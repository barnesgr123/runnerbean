function [z0,zsubj,nativemeshnames] = sample_brains_grb_rand0(Nb, PC,mz, sdz, output_folder, template_folder,template_meshfiles)
% FORMAT sample_brains([Nb], [output_folder], [PC])
%   N - Number of samples {1}
%   output_folder - Folder where to write generated gifti files {'.'}
%   pc - Shoot along these principal components. If NaN, random sample. {NaN}
%
% /!\ A few paths are hard-coded and need to be adapted.

%% some PC indices (eg -2) can be negative meaning their abs components (2) will be linearly shifted in opposite direction
%% Load data

% -------------------------------------------------------------------------
% Specify model files
if nargin<6,
    template_meshfiles=[];
end;
if isempty(template_meshfiles),
    template_meshfiles{1}=[output_folder 'mesh_cortex_template.gii'];
end;

fsubspace = [template_folder filesep 'subspace_scaled.nii'];
faffine   = [output_folder filesep 'affine.mat'];
mkdir([output_folder filesep 'Cerebros']);

% Subject components
fcode = [output_folder filesep 'latent_code.mat'];
load(fcode, 'z');						% JD: All PCs come from Subject's brain
zsubj=z; %% return original brain shape


% -------------------------------------------------------------------------
% Read model files
for f=1:numel(template_meshfiles), %% maybe more than 1 mesh eg cortex+ inner skull
    cortex{f}   = gifti([output_folder filesep template_meshfiles{f}]);
end;
subspace = nifti(fsubspace);
load(faffine, 'tpl2native');

%% Define way of randomisation
% - Creating a linearly spaced vector between [-sp sp] around z (subject) or zero (canonical)
% - Gaussian random distribution with mean over z (subject) or zero (canonical)

Npc = length(PC);
absPC=abs(PC); % some components have negative values indicated we want to shift in opposite direction
sgnPC=sign(PC); %% + or -1 depending on how to shift
if Npc == 1
    warning('moving one component systematically ')
    z0 = linspace(-sp,sp,Nb);				% Canonical
else
%     fprintf('Making random transformation from uniform distributionof %3.2f sds',sp)
% 
%     z0=-sp+2*rand(Npc,Nb)*sp;
%     z0(:,1)=zeros(Npc,1); %% get first brian as close as we can
%     fprintf('Linear spacing of lots of components..')
%     mz = linspace(-sp,sp,Nb);				% Canonical mean
%     sdz=mean(diff(mz))*2; %% canonical spread of distribution
    
    mz0=ones(Npc,Nb).*mz; %% mean points
    mz0=mz0+(rand(Npc,Nb)-0.5)*sdz; %% uniform random about that mean
    z0=mz0.*repmat(sgnPC',1,Nb); %% random signs at each component
end;
% % 		z0 = zeros(N,1);
% % 		z0(1:N-1)	= - sp + 2*sp*rand(N-1,1);				% Uniform distribution [-sp, sp] around 0
% % 		z0(N)		= z(pc);								% Ground truth
% 	else	% For distorting several PCs at the same time
% z0 = repmat(z,1,Nb);
% for i1 = 1:Npc
%     %z0(PC(i1),1:Nb) = z(PC(i1)); % - sp + 2*sp*rand(Nb,1);	% Uniform distribution [-sp, sp] around z(pc)
%     z0(PC(i1),1:Nb) = z(PC(i1)) - sp + 2*sp*rand(Nb,1);	% Uniform distribution [-sp, sp] around z(pc)
% end



% -------------------------------------------------------------------------
% Loop about samples
fprintf('Sample brain ');
for n=1:Nb
    if n > 1
        fprintf(repmat('\b', [1 4+numel(num2str(n-1))]));
    end
    fprintf('%d ...', n);

    % ---------------------------------------------------------------------
    % Sample transformation [template -> random]

    % Shoot along principal component
    z = zeros(subspace.dat.dim(6),1);
    z(absPC) = z0(:,n);


    iy = sample_deformation(z, fsubspace);          % voxel-to-voxel
    iy = warps('compose', subspace.mat, iy);        % voxel-to-mm

    % ---------------------------------------------------------------------
    % Deform mesh
    for j=1:numel(template_meshfiles),
        [a1,b1,c1]=fileparts(template_meshfiles{j});
        namestr=[b1(1:findstr(b1,'template')-1) 'native']; %% output will be in native (MRI) space
        ncortex=cortex{j};
        randomcortex = transform_mesh(ncortex, iy, subspace.mat);

        % ---------------------------------------------------------------------
        % Send to Gareth space
        randomcortex = transform_mesh(randomcortex, tpl2native);

        % ---------------------------------------------------------------------
        % Write on disk
        nativemeshnames{n,j}=[output_folder filesep 'Cerebros\' sprintf('smp_%d_%s.gii',n,namestr)];
        save(randomcortex,nativemeshnames{n,j});
    end; % for j
end; % for n
fprintf('\n');


