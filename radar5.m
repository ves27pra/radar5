%% Experiment: Pulse Doppler Radar Target Detection
% Aim: Estimate target RANGE and VELOCITY using Pulse Doppler Radar
% Steps: Generate pulse train -> Simulate multi-target echoes ->
% Matched filtering (Range compression) -> Doppler FFT ->
% Range-Doppler Map -> Peak detection -> Range & Velocity estimate
%
% Works on plain MATLAB / Octave (no Phased Array Toolbox functions used,
% so it also runs on older MATLAB versions).

clear; clc; close all;

%% 1. Radar System Parameters
c = 3e8; % Speed of light (m/s)
fc = 10e9; % Carrier frequency (Hz) -> X-band
PRF = 5000; % Pulse Repetition Frequency (Hz)
PRI = 1/PRF; % Pulse Repetition Interval (s)
tau = 2e-6x; % Pulse width (s)
fs = 20e6; % Sampling frequency (Hz)
Np = 128; % Number of pulses (slow-time samples)

lambda = c/fc; % Wavelength
Ns = round(PRI*fs); % Samples per PRI (fast-time samples)
t_fast = (0:Ns-1)/fs; % Fast-time vector (within one PRI)

Rmax = c*PRI/2; % Unambiguous max range
Vmax = lambda*PRF/4; % Unambiguous max (radial) velocity

fprintf('Unambiguous Max Range : %.2f m\n', Rmax);
fprintf('Unambiguous Max Velocity: %.2f m/s\n', Vmax);

%% 2. Generate Transmitted Pulse Train
pulse_samples = round(tau*fs);
tx_pulse = ones(1, pulse_samples); % Simple rectangular pulse
tx_pulse_full = zeros(1, Ns);
tx_pulse_full(1:pulse_samples) = tx_pulse; % One PRI of the tx waveform

%% 3. Define Multiple Moving Targets
% Each target: [Range (m), Radial Velocity (m/s), Amplitude]
targets = [ 3000, 20, 1.0; % Target 1: 3 km, moving away at 20 m/s
            7000, -35, 0.8; % Target 2: 7 km, approaching at 35 m/s
           12000, 10, 0.6]; % Target 3: 12 km, moving away at 10 m/s

numTargets = size(targets,1);

%% 4. Simulate Received Echo Matrix (Fast-time x Slow-time)
rxMatrix = zeros(Ns, Np); % rows = fast time (range), cols = pulse #
noise_std = 0.05;

for p = 1:Np % loop over pulses (slow time)
    echo = zeros(1, Ns);
    for k = 1:numTargets
        R0 = targets(k,1);
        v = targets(k,2);
        A = targets(k,3);

        % Range at the time of this pulse (target moves slightly each PRI)
        R_p = R0 + v*(p-1)*PRI;

        % Round-trip delay for this pulse
        delay_samples = round(2*R_p/c * fs);

        % Doppler phase shift accumulated across pulses
        doppler_phase = exp(1j*4*pi*v*(p-1)*PRI/lambda);

        if delay_samples < Ns
            idx = (1:pulse_samples) + delay_samples;
            idx = idx(idx <= Ns);
            echo(idx) = echo(idx) + A*doppler_phase*tx_pulse(1:length(idx));
        end
    end
    % Add complex Gaussian noise
    noise = noise_std*(randn(1,Ns)+1j*randn(1,Ns));
    rxMatrix(:,p) = (echo + noise).';
end

%% 5. Matched Filtering (Pulse / Range Compression)
matchedFilter = conj(fliplr(tx_pulse));
rangeCompressed = zeros(Ns, Np);
for p = 1:Np
    mf_out = conv(rxMatrix(:,p), matchedFilter, 'same');
    rangeCompressed(:,p) = mf_out;
end

%% 6. Doppler Processing (FFT across slow time / pulses)
% Window each range bin's slow-time samples to reduce sidelobes
win = hamming(Np).';
rangeDopplerRaw = rangeCompressed .* win;

% 2-D FFT: FFT already applied conceptually along range (matched filter);
% here we take FFT along the Doppler (pulse) dimension for every range bin.
RangeDopplerMap = fftshift(fft(rangeDopplerRaw, Np, 2), 2);
RD_mag = abs(RangeDopplerMap);
RD_dB = 20*log10(RD_mag./max(RD_mag(:)) + eps);

%% 7. Axes for the Range-Doppler Map
rangeAxis = (0:Ns-1)*c/(2*fs); % meters
dopplerFreqAxis = (-Np/2:Np/2-1)*(PRF/Np); % Hz
velocityAxis = dopplerFreqAxis*lambda/2; % m/s

%% 8. Plot Range-Doppler Map
figure('Name','Range-Doppler Map');
imagesc(velocityAxis, rangeAxis, RD_dB);
axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('Range-Doppler Map (Pulse Doppler Radar)');
colorbar;
caxis([-40 0]);
ylim([0 15000]);

figure('Name','Range-Doppler Surface');
mesh(velocityAxis, rangeAxis, RD_dB);
xlabel('Velocity (m/s)');
ylabel('Range (m)');
zlabel('Magnitude (dB)');
title('Range-Doppler Surface Plot');
zlim([-40 0]);
ylim([0 15000]);

%% 9. Peak Detection: Estimate Range & Velocity of Each Target
fprintf('\n--- Detected Targets ---\n');
RD_mag_search = RD_mag;
RD_mag_search(1:round(0.5*Ns/fs*c/2/(rangeAxis(2)-rangeAxis(1))),:) = 0; %#ok<NASGU>

detected = 0;
threshold = 0.4*max(RD_mag(:));
tempMap = RD_mag;

for n = 1:numTargets
    [maxVal, linIdx] = max(tempMap(:));
    if maxVal < threshold
        break;
    end
    [rIdx, dIdx] = ind2sub(size(tempMap), linIdx);
    estRange = rangeAxis(rIdx);
    estVel = velocityAxis(dIdx);
    detected = detected + 1;
    fprintf('Target %d -> Estimated Range: %8.1f m | Estimated Velocity: %7.2f m/s\n', ...
        detected, estRange, estVel);

    % Null out a small neighborhood around this peak so the next
    % iteration finds a different target
    rWin = max(1,rIdx-5):min(Ns,rIdx+5);
    dWin = max(1,dIdx-3):min(Np,dIdx+3);
    tempMap(rWin, dWin) = 0;
end

fprintf('\n--- Ground Truth (for comparison) ---\n');
for k = 1:numTargets
    fprintf('Target %d -> Range: %8.1f m | Velocity: %7.2f m/s\n', ...
        k, targets(k,1), targets(k,2));
end