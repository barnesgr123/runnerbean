%simdipmesh_JD
%
% Created by:	Jose David Lopez - ralph82co@gmail.com
%
% Date: May 2012


function [D,meshsourceind,signal] = simdipmesh_JD(D)

if isfield(D.inv{1},'inverse')
	D.inv{1} = rmfield(D.inv{1},'inverse');
end

modalities	= D.inv{1}.forward.modality;		% MEG in this case
Ic			= indchantype(D, modalities, 'GOOD');


%% GENERATION OF NEURAL SOURCES
% SPM works in mm, lead fields are correctly calculated in SI
% units (spm_cond_units)
% Select number of sources

Ndip		= 1;			% Always select number of sources

randvertind	= randi(size(D.inv{1}.mesh.tess_mni.vert,1),Ndip,1);		% Eleccion de vértices de fuentes al azar
% randvertind = 1075;%3608%der: 6628 - 7972

patchmni	= D.inv{1}.mesh.tess_mni.vert(randvertind,:);				% Posición de dichas fuentes

meshsourceind = zeros(1,Ndip);
for d = 1:Ndip
	vdist	= D.inv{1}.mesh.tess_mni.vert - repmat(patchmni(d,:),size(D.inv{1}.mesh.tess_mni.vert,1),1);
	dist	= sqrt(dot(vdist',vdist'));
	[~,meshsourceind(d)] = min(dist);
end

dipfreq		= 20; % randi([1 20],1,Ndip);				% Source frequency %% GRB change
dipamp		= ones(Ndip,1).*1000*1e-9;			% Source amplitude in nAm


%% WAVEFORM FOR EACH SOURCE

Ntrials = D.ntrials;				% Number of trials

% define period over which dipoles are active
startf1  = 0;					% (sec) start time
duration = 5;					% duration 

endf1 = duration + startf1;
f1ind = intersect(find(D.time>startf1),find(D.time<=endf1));

% Create the waveform for each source
Nn		= length(D.time);
signal	= zeros(Ndip,Nn);
for j = 1:Ndip					% For each source
	for i = 1:Ntrials			% and trial
		f1		= dipfreq(j);	% Frequency depends on stim condition
		amp1	= dipamp(j);	% also the amplitude
		phase1	= pi/2;
		signal(j,f1ind) = signal(j,f1ind)...
			+ amp1*sin((D.time(f1ind)...
			- D.time(min(f1ind)))*f1*2*pi + phase1);
	end
end


%% CREATE A NEW LEAD FIELD AND SMOOTHER

fprintf('Computing Gain Matrix: ')
[L,D] = spm_eeg_lgainmat(D);				% Gain matrix
Nd    = size(L,2);							% number of dipoles
X	  = zeros(Nd,Nn);						% Matrix of dipoles
fprintf(' - done\n')

% Green function for smoothing sources with the same distribution than SPM8
fprintf('Computing Green function from graph Laplacian:')
vert  = D.inv{1}.mesh.tess_mni.vert;
face  = D.inv{1}.mesh.tess_mni.face;
A     = spm_mesh_distmtx(struct('vertices',vert,'faces',face),0);

GL    = A - spdiags(sum(A,2),0,Nd,Nd);
GL    = GL*0.6/2;
Qi    = speye(Nd,Nd);
QG    = sparse(Nd,Nd);
for i = 1:8
    QG = QG + Qi;
    Qi = Qi*GL/i;
end
QG    = QG.*(QG > exp(-8));
QG    = QG*QG;
SMTH_OFF=1;
if SMTH_OFF
    warning('No smoothing')
    QG=speye(size(QG));
end;
clear Qi A GL
fprintf(' - done\n')

%% PERFORM THE FORWARD PROBLEM

fprintf('Creating the dataset:')

% Add waveform of all smoothed sources to their equivalent dipoles
% QGs add up to 0.9854
for j=1:Ndip 
	for i=1:Nn
		X(:,i) = X(:,i) + signal(j,i)*QG(:,meshsourceind(j));
	end
end

D(Ic,:,1) = L*X;					% Forward problem

% Copy data to all trials (It should not be the same, of course)
for i=1:Ntrials
	D(:,:,i) = D(:,:,1);
end

fprintf(' - done\n')

%% ADD WHITE NOISE
% In decibels

SNR				= 10;		% Select SNR in dB
fprintf('Adding white noise with SNR= %2idB',SNR)

allchanstd		= (std(D(Ic,:,1)'));
meanrmssignal	= mean(allchanstd);
MEANCHANNOISE	= 1; % TAKE MEAN RMS NOISE RATHER THAN A DIFFERENT NOISE LEVEL PER CHANNEL
for i = 1:size(Ic,1)
	if Ndip>0
		if ~MEANCHANNOISE
            channoise = std(D(Ic(i),:,1)) .* randn(size(D(Ic(i),:,1)))/(10^(SNR/20));
		else
            channoise = meanrmssignal .* randn(size(D(Ic(i),:,1)))/(10^(SNR/20));
		end
	else
        channoise = randn(size(D(Ic(i),:,1))); %% just add noise when no dips specifed
	end
	%allchannoise(i,:)=channoise;
	D(Ic(i),:,1) = D(Ic(i),:,1) + channoise;
end

%% Plot and save

% graf_SPM_JD(D,X);
% 
% figure
% hold on
% aux = L(1,:)*X;
% plot(D.time,aux,'r');
% hold on
% plot(D.time,D(Ic(1),:,1));
% title('Measured activity over MLC11');
% legend('Noiseless','Noisy');

D.save;

fprintf('\n Finish\n')

