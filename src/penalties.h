#pragma once

#include <vector>

struct iterative_params{
    size_t Nx;
    size_t Ny;
    size_t Nz;
    double alpha;
    double beta;
    double lambda;
    double delta;
    int num_neighbors;
    double * weights;
    double weights_scale;
};

void   initialize_2d_weights(struct iterative_params * ip);
double quadratic(int curr_idx,struct iterative_params * ip      , float * recon_volume);
double edge_preserving(int curr_idx,struct iterative_params * ip, float * recon_volume);

