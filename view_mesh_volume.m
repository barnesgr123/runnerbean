function view_mesh_volume(mesh, volume, mat)
% FORMAT view_mesh_volume(mesh, volume, mat)
%   mesh   - gifti object
%   volume - nd-array or nifti volume
%   mat    - voxel-to-world matrix of the volume

if nargin < 3
    if nargin >= 2
        if isa(volume, 'nifti')
            mat = volume.mat;
        else
            mat = eye(4);
        end
    end
end

% -------------------------------------------------------------------------
% Plot mesh
% -------------------------------------------------------------------------
figure
mesh.vertices = mesh.mat(1:3,:) * ...
    [mesh.vertices';ones(1,size(mesh.vertices,1))];
patch('vertices',mesh.vertices','faces',mesh.faces,...
    'EdgeColor',[1 .7 .8],'FaceColor','none')

% -------------------------------------------------------------------------
% Plot volume
% -------------------------------------------------------------------------
if nargin >= 2
    if isa(volume, 'nifti')
        volume = volume.dat;
    end
    mri      = struct;
    mri.dat  = volume;
    mri.mat  = mat;

    hold on
    pls = 0.05:0.2:0.9;
    d   = size(mri.dat);
    pls = round(pls.*d(3));
    for i=1:numel(pls)
        [x,y,z] = ndgrid(1:d(1),1:d(2),pls(i));
        f1 = mri.dat(:,:,pls(i));
        M  = mri.mat;
        x1 = M(1,1)*x+M(1,2)*y+M(1,3)*z+M(1,4);
        y1 = M(2,1)*x+M(2,2)*y+M(2,3)*z+M(2,4);
        z1 = M(3,1)*x+M(3,2)*y+M(3,3)*z+M(3,4);
        s  = surf(x1,y1,z1,f1);
        set(s,'EdgeColor','none')
    end
    axis image off;
    colormap('gray');
end

view(-135,45);
rotate3d on