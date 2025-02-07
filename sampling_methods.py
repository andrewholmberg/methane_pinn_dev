import numpy as np
def spread_sample_source(source_locs, source_quant, radii, n_outer, max_iters, current_iter,t_max, t_points):
    source_locs = np.array(source_locs)
    ls = np.empty((0,3),dtype=np.float64)
    for iter in range(current_iter+1):
        iter_prop = iter/max_iters
        radii_temp = np.array(radii)*iter_prop
        ls=np.concatenate((ls,sample_points_on_ellipsoid(radii_temp,source_locs[2],n_outer*iter//max_iters+1)))
    # print(len(ls),len(np.tile(ls,t_points)))
    return np.column_stack([np.tile(np.linspace(0,t_max,t_points),len(ls)),np.tile(ls,(t_points,1))])


def sample_points_on_ellipsoid(radii, source_loc,  n_points):
    # Generate uniform spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # Azimuthal angle
    phi = np.arccos(np.random.uniform(-1, 1, n_points))  # Polar angle
    
    # Convert spherical coordinates to Cartesian coordinates on the unit sphere
    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)
    
    # Scale the unit sphere to the ellipsoid using the given radii
    x_ellipsoid = radii[0] * x_sphere
    y_ellipsoid = radii[1] * y_sphere
    z_ellipsoid = radii[2] * z_sphere
    return  np.column_stack(np.array([x_ellipsoid+source_loc[0], y_ellipsoid+source_loc[1], z_ellipsoid+source_loc[2]]))

# source_locs = [[50,50,2]]
# radii = [30,30,1]
# source_quant = [1]
# x = spread_sample_source(source_locs, source_quant, radii, 100,20,20,10,10)
# print(x)
# print(len(x))