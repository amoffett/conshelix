import mdtraj as md
import numpy as np

def helix_CVs(traj1,traj2):
    """
    Calculate the minimum distance and crossing angle between the principal axes of two helices.

    Parameters
    ----------
    traj1, traj2: mdtraj trajectories
        Trajectories of helix alpha carbon atoms to analyze.

    Returns
    -------
    D: ndarray
        Minimum distance (nm) between the principal axes for each frame.
    theta: ndarray
        Crossing angle (rad) between the two principal axes for each frame.

    References:
    [1] Lee, J and Im, W. J. Comput. Chem. (28)669-680, 2007.
    """
    I_1 = md.geometry.compute_inertia_tensor(traj1)
    I_2 = md.geometry.compute_inertia_tensor(traj2)
    a_1 = []
    a_2 = []
    for i in I_1:
        e, v = np.linalg.eig(i)
        vec = v[:,np.argmin(e)]
        if vec[-1] < 0:
            vec = -1 * vec
        a_1.append(vec)
    for i in I_2:
        e, v = np.linalg.eig(i)
        vec = v[:,np.argmin(e)] 
        if vec[-1] < 0:
            vec = -1 * vec
        a_2.append(vec)
    a_1 = np.vstack(a_1)
    a_2 = np.vstack(a_2)
    r1_1 = traj1.xyz[:,0,:]  
    rn_1 = traj1.xyz[:,-1,:]
    r_mean_1 = np.mean(traj1.xyz,axis=1)
    r1_2 = traj2.xyz[:,0,:]
    rn_2 = traj2.xyz[:,-1,:]
    r_mean_2 = np.mean(traj2.xyz,axis=1)
    b_1 = r_mean_1 + np.sum(a_1*(r1_1-r_mean_1),axis=1,keepdims=True)*a_1
    e_1 = r_mean_1 + np.sum(a_1*(rn_1-r_mean_1),axis=1,keepdims=True)*a_1
    b_2 = r_mean_2 + np.sum(a_2*(r1_2-r_mean_2),axis=1,keepdims=True)*a_2
    e_2 = r_mean_2 + np.sum(a_2*(rn_2-r_mean_2),axis=1,keepdims=True)*a_2
    W_1 = np.sum((b_1-b_2)*(e_1-b_1),axis=1)
    W_2 = np.sum((b_1-b_2)*(e_2-b_2),axis=1)
    U_11 = np.sum((e_1-b_1)*(e_1-b_1),axis=1)
    U_12 = np.sum((e_1-b_1)*(e_2-b_2),axis=1)
    U_21 = -U_12
    U_22 = np.sum((e_2-b_2)*(e_2-b_2),axis=1)
    Det = U_11*U_22-U_12**2
    S_1 = (W_2*U_12-W_1*U_22)/Det
    for i in range(S_1.shape[0]):
        if S_1[i] < 0:
            S_1[i] = 0
        elif S_1[i] > 1:
            S_1[i] = 1
    S_1 = S_1.reshape(S_1.shape[0],1)
    S_2 = (W_2*U_11-W_1*U_12)/Det
    for i in range(S_2.shape[0]):
        if S_2[i] < 0:
            S_2[i] = 0
        elif S_2[i] > 1:
            S_2[i] = 1
    S_2 = S_2.reshape(S_2.shape[0],1)
    t_1 = b_1 + S_1 * (e_1 - b_1)
    t_2 = b_2 + S_2 * (e_2 - b_2)
    D = np.sqrt(np.sum((t_1 - t_2)**2,axis=1,keepdims=True))
    h = (t_2 - t_1) / D
    l = a_1
    m = -a_2
    arg_theta = np.sum(np.cross(l,h)*np.cross(h,m),axis=1,keepdims=True)/(np.sqrt(np.sum(np.cross(l,h)**2,axis=1,keepdims=True))*np.sqrt(np.sum(np.cross(h,m)**2,axis=1,keepdims=True)))
    angle_sign = np.sign(np.sum((np.cross(np.cross(l,h),np.cross(h,m)))*h,axis=1,keepdims=True))
    theta = np.arccos(arg_theta) * angle_sign
    return D, theta
