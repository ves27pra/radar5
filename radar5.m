clear; clc; close all;

c = 3e8; fc = 10e9;
PRF = 5000; PRI = 1/PRF;
tau = 2e-6; fs = 20e6; Np = 128;
lambda = c/fc;
Ns = round(PRI*fs);

Rmax = c*PRI/2;
Vmax = lambda*PRF/4;
fprintf('Max Range: %.2f m, Max Velocity: %.2f m/s\n', Rmax, Vmax);

ps = round(tau*fs);
tx = ones(1, ps);

targets = [3000  20  1.0;
           7000 -35  0.8;
          12000  10  0.6];
nT = size(targets,1);

rx = zeros(Ns, Np);
for p = 1:Np
    echo = zeros(1, Ns);
    for k = 1:nT
        R = targets(k,1) + targets(k,2)*(p-1)*PRI;
        d = round(2*R/c * fs);
        ph = exp(1j*4*pi*targets(k,2)*(p-1)*PRI/lambda);
        if d < Ns
            idx = (1:ps) + d;
            idx = idx(idx <= Ns);
            echo(idx) = echo(idx) + targets(k,3)*ph*tx(1:length(idx));
        end
    end
    rx(:,p) = echo.' + 0.05*(randn(Ns,1) + 1j*randn(Ns,1));
end

mf = conj(fliplr(tx));
rc = zeros(Ns, Np);
for p = 1:Np
    rc(:,p) = conv(rx(:,p), mf, 'same');
end

win = hamming(Np).';
RD = fftshift(fft(rc .* win, Np, 2), 2);
RDmag = abs(RD);
RDdB = 20*log10(RDmag/max(RDmag(:)) + eps);

rAxis = (0:Ns-1)*c/(2*fs);
vAxis = (-Np/2:Np/2-1)*(PRF/Np)*lambda/2;

figure;
imagesc(vAxis, rAxis, RDdB); axis xy;
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title('Range-Doppler Map'); colorbar;
caxis([-40 0]); ylim([0 15000]);

figure;
mesh(vAxis, rAxis, RDdB);
xlabel('Velocity (m/s)'); ylabel('Range (m)'); zlabel('dB');
title('Range-Doppler Surface');
zlim([-40 0]); ylim([0 15000]);

fprintf('\n--- Detected Targets ---\n');
tmp = RDmag;
thr = 0.4*max(RDmag(:));
for n = 1:nT
    [v, i] = max(tmp(:));
    if v < thr, break; end
    [r, d] = ind2sub(size(tmp), i);
    fprintf('Target %d -> Range: %.1f m | Velocity: %.2f m/s\n', n, rAxis(r), vAxis(d));
    tmp(max(1,r-5):min(Ns,r+5), max(1,d-3):min(Np,d+3)) = 0;
end

fprintf('\n--- Ground Truth ---\n');
for k = 1:nT
    fprintf('Target %d -> Range: %.1f m | Velocity: %.2f m/s\n', k, targets(k,1), targets(k,2));
end
