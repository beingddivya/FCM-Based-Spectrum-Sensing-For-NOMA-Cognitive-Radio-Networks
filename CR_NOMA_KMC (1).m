% Uplink NOMA spectrum senisng: Using KMC
% SM: 22-02-2024
clear;
clc;
close all;
SNRdB = 6;
SNR = 10^(SNRdB/10);
noiseVar = 1;

M = 15;                 % Number of secondary users
N = 100;                % Time slots in sensing interval
K = 1600;               % Training data length
MaxTrials = 100;        % Number of trials of Monte Carlo simulations with different channels
Omega = [2,1];          % Power ratios of PU
Omega = Omega/sum(Omega);
Q = length(Omega);      % Number of NOMA primary users

data = zeros(K,M);
accu = 0;

for ep = 1:MaxTrials
    Theta = rand(K,Q)>0.5;  % State of the PUs
    H = (randn(Q,M) + 1i*randn(Q,M))/sqrt(2); % Rayleigh channel between PU and SU
    H = H./sqrt(sum(abs(H).^2,2));
    signalPower = SNR*noiseVar/((Omega(1)*sum(abs(H(1,:)).^2)/M)+(Omega(2)*sum(abs(H(2,:)).^2)/M));

    for k = 1:K
        theta = Theta(k,:);
        S = sqrt(signalPower)*sqrt(Omega).*(randn(N,Q) + 1i*randn(N,Q))/sqrt(2);   % PU signal

        signalUser = (Theta(k,:).*S)*H;
        noiseUser = sqrt(noiseVar)*(randn(N,M) + 1i*randn(N,M))/sqrt(2);
        rxUser = signalUser + noiseUser;

        signal_power = sum(abs(S*H).^2);
        noise_power = sum(abs(noiseUser).^2);
        snr_actual = 10*log10(sum(signal_power./noise_power)/M);
        u = sum(abs(rxUser).^2)/N;
        data(k,:) = u;
    end
    X = data;
    %% -------------------------- KMC method ------------------------------
    [pm,MU] = kmeans(X,4);
    norm_MU = sqrt(sum(MU.^2,2));
    [~,idx] = sort(norm_MU,'ascend');
    P(find(pm==idx(1))) = 0;
    P(find(pm==idx(2))) = 1;
    P(find(pm==idx(3))) = 2;
    P(find(pm==idx(4))) = 3;
    detected_state = P';
    actual_state = binaryVectorToDecimal(Theta);
    accu = accu + sum(detected_state==actual_state);
end
accuracy = accu/(MaxTrials*K)




