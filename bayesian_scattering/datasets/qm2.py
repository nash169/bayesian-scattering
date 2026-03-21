import torch
import numpy as np
import pickle
from importlib.resources import files
from skimage.transform import AffineTransform, warp, SimilarityTransform, resize, rescale
from bayesian_scattering.datasets.abstract_dataset import AbstractDataset


class QM2(AbstractDataset):
    def __init__(
            self,
            density_type='dirac',
            store_path=None,
            shard_size: int = 5_000,
            dtype=np.float32,
            unimol: bool = False,
            **kwargs
    ):
        self.density_type = density_type

        with open(files('rsc').joinpath('qm7_2d.pkl'), 'rb') as f:
            data = pickle.load(f)

        self.coordinates = 0.529177 * data['atoms_coordinate']
        self.atomic_charge = data['atomic_charge']
        self.labels = np.squeeze(data['atomization_energies'])

        with open(files('rsc').joinpath('atom_rho.pkl'), 'rb') as f:
            self.atom_rho = pickle.load(f)

        self.unimol = unimol
        self.atomic_number = {
            1: 'H',
            6: 'C',
            7: 'N',
            8: 'O',
            16: 'S',
            17: 'Cl'
        }

        super().__init__(
            cache_root=store_path,
            dataset_id=f"qm2_{self.density_type}",
            shard_size=shard_size,
            dtype=dtype
        )

    def __getitem__(self, idx: int):
        if not self.unimol:
            return super().__getitem__(idx)
        return self.unimol_dict(idx), self.labels[idx]

    def generate_sample(self, idx) -> torch.Tensor:
        coordinate, atomic_charge, atomic_rho = self.coordinates[idx], self.atomic_charge[idx], self.atom_rho

        src_box = [-2.5, 2.5, -2.5, 2.5]
        src_delta = [src_box[1] - src_box[0], src_box[3] - src_box[2]]
        src_res = [501, 501]

        # dest_box = [-9, 9, -7, 7]
        dest_box = [-9, 9, -9, 9]
        dest_delta = [dest_box[1] - dest_box[0], dest_box[3] - dest_box[2]]
        # dest_res = [2**9, round((dest_delta[1] / dest_delta[0]) * 2**9)]
        dest_res = [2**9, 2**9]

        x = np.linspace(dest_box[0], dest_box[1], dest_res[0])
        y = np.linspace(dest_box[2], dest_box[3], dest_res[1])

        atom_list = ['Hydrogen', 'Carbon', 'Nitrogen', 'Oxygen', 'Sulfur', 'Chlorine']
        Z_atom = np.array([1, 6, 7, 8, 16, 17])
        Z_atom_cor = np.array([0, 2, 2, 2, 10, 10])
        Z_atom_val = np.array([1, 4, 5, 6, 6, 7])

        valid_atoms = atomic_charge > 0
        Z = atomic_charge[valid_atoms]
        R = coordinate[valid_atoms, :]

        rho = np.zeros(dest_res)

        for j in range(R.shape[0]):
            Z_ind = np.where(Z_atom == Z[j])[0][0]
            pos = R[j, :]

            if self.density_type == 'dirac':
                x_ind = np.argmin(np.abs(x - pos[0]))
                y_ind = np.argmin(np.abs(y - pos[1]))
                rho_atom_translate = np.zeros(dest_res)
                rho_atom_translate[x_ind, y_ind] = Z_atom[Z_ind]
                I = np.trapz(np.trapz(rho_atom_translate, y, axis=1), x)
                rho_atom_translate = (Z_atom[Z_ind] / I) * rho_atom_translate
                rho += rho_atom_translate
            else:
                pos = R[j, :]
                dx = pos[0] / src_delta[0] * src_res[0]
                dy = pos[1] / src_delta[1] * src_res[1]
                translate = AffineTransform(translation=(dy, dx))
                scaling = AffineTransform(scale=(src_delta[1] / dest_delta[1], src_delta[0] / dest_delta[0]))
                center_x = src_box[0] + src_delta[0]**2 / dest_delta[0] / 2
                center_y = src_box[2] + src_delta[1]**2 / dest_delta[1] / 2
                center = AffineTransform(translation=(-(center_y / src_delta[1]) * src_res[1], -(center_x / src_delta[0]) * src_res[0]))

                if self.density_type == 'atomic':
                    rho_atom_translate = warp(atomic_rho[atom_list[Z_ind]]['atomic'], (translate + scaling +
                                              center).inverse, output_shape=(src_res[0], src_res[1]))
                    rho_atom_translate = resize(rho_atom_translate, output_shape=(dest_res[0], dest_res[1]))
                    I = np.trapz(np.trapz(rho_atom_translate, y, axis=1), x)
                    rho_atom_translate = (Z_atom[Z_ind] / I) * rho_atom_translate
                    rho += rho_atom_translate
                elif self.density_type == 'valence':
                    rho_atom_translate = warp(atomic_rho[atom_list[Z_ind]]['valence'], (translate +
                                              scaling + center).inverse, output_shape=(src_res[0], src_res[1]))
                    rho_atom_translate = resize(rho_atom_translate, output_shape=(dest_res[0], dest_res[1]))
                    I = np.trapz(np.trapz(rho_atom_translate, y, axis=1), x)
                    rho_atom_translate = (Z_atom_val[Z_ind] / I) * rho_atom_translate
                    rho += rho_atom_translate
                elif self.density_type == 'core':
                    if Z_atom_cor[Z_ind] > 0:
                        rho_atom_translate = warp(atomic_rho[atom_list[Z_ind]]['core'], (translate + scaling +
                                                  center).inverse, output_shape=(src_res[0], src_res[1]))
                        rho_atom_translate = resize(rho_atom_translate, output_shape=(dest_res[0], dest_res[1]))
                        I = np.trapz(np.trapz(rho_atom_translate, y, axis=1), x)
                        rho_atom_translate = (Z_atom_cor[Z_ind] / I) * rho_atom_translate
                        rho += rho_atom_translate

        return torch.from_numpy(rho).float()

    def unimol_dict(self, idx):
        coordinates = np.append(
            self.coordinates,
            np.zeros((*(self.coordinates.shape[:-1]), 1)),
            axis=2
        )
        formatted_atoms = [self.atomic_number[int(z)] for z in self.atomic_charge[idx] if z > 0]
        formatted_coords = coordinates[idx][self.atomic_charge[idx] != 0].tolist()
        return {'atoms': formatted_atoms, 'coordinates': formatted_coords}

    def __len__(self):
        return len(self.labels)

    @property
    def shape(self) -> torch.Size:
        return torch.Size([512, 512])

    @property
    def channels(self) -> int:
        return 1

    def to(self, device):
        return self
