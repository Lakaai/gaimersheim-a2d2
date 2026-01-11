import numpy as np

def dynamics(x: np.ndarray, jacobian: bool = False) -> np.ndarray:
    """Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw."""

    # 
    # dnu/dt  =          0 + dwnu/dt
    # deta/dt = JK(eta)*nu +       0
    # dm/dt   =          0 +       0
    # \_____/   \________/   \_____/
    #  dx/dt  =    f(x)    +  dw/dt
    # 
    #        [          0 ]
    # f(x) = [ JK(eta)*nu ]
    #        [          0 ] for all map states
    # 
    #        [                    0 ]
    #        [                    0 ]
    # f(x) = [    Rnb(thetanb)*vBNb ]
    #        [ TK(thetanb)*omegaBNb ]
    #        [                    0 ] for all map states
    # 

    # JK(eta) = [Rnb(thetanb) 0; 0 TK(thetanb)]

    f = np.zeros_like(x) # TODO Investigate what this function does. 

    # Extract current state
    vBNb = x[0:3]      # Body translational velocity
    omegaBNb = x[3:6]  # Body angular velocity
    rBNn = x[6:9]      # Body position
    Thetanb = x[9:12]  # Body orientation

    # Compute rotation matrix from body to navigation frame
    Rnb = rpy2rot(Thetanb)

    # Update position
    f[6:9] = Rnb @ vBNb
    
    # Update orientation
    T = np.eye(3)
    phi = Thetanb[0]
    theta = Thetanb[1]
    T[0,1] = np.sin(phi) * np.tan(theta)
    T[0,2] = np.cos(phi) * np.tan(theta)
    T[1,1] = np.cos(phi)
    T[1,2] = -np.sin(phi)
    T[2,1] = np.sin(phi) / np.cos(theta)
    T[2,2] = np.cos(phi) / np.cos(theta)
    f[9:12] = T @ omegaBNb

    # Velocities and landmark states remain constant in this model
    # f[0:3] = [0, 0, 0]  # dvBNb/dt = 0
    # f[3:6] = [0, 0, 0]  # domegaBNb/dt = 0
    # f[12:] = 0          # dm/dt = 0 for all landmarks

    if jacobian:
        # Jacobian J = df/dx
        #     
        #     [  0                  0 0 ]
        # J = [ JK d(JK(eta)*nu)/deta 0 ]
        #     [  0                  0 0 ]
        #
        J = np.zeros((f.size, x.size))

        # Jacobian for position update (df_r / dv)
        J[6:9, 0:3] = Rnb 
        
        # Jacobian for position update (df_r / dTheta)
        psi = Thetanb[2]
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        ttheta = np.tan(theta)

        dRnb_dphi = np.array([
            [0, sphi*spsi + cphi*stheta*cpsi, cphi*spsi - sphi*stheta*cpsi],
            [0, sphi*cpsi - cphi*stheta*spsi, cphi*cpsi + sphi*stheta*spsi],
            [0, cphi*ctheta, -sphi*ctheta]
        ])

        dRnb_dtheta = np.array([
            [-ctheta*cpsi, -stheta*sphi*cpsi, -stheta*cphi*cpsi],
            [-ctheta*spsi, -stheta*sphi*spsi, -stheta*cphi*spsi],
            [-stheta, ctheta*sphi, ctheta*cphi]
        ])

        dRnb_dpsi = np.array([
            [-stheta*spsi, cphi*cpsi - sphi*stheta*spsi, sphi*cpsi + cphi*stheta*spsi],
            [stheta*cpsi, -cphi*spsi - sphi*stheta*cpsi, -sphi*spsi + cphi*stheta*cpsi],
            [0, 0, 0]
        ])
        J[6:9, 9] = dRnb_dphi @ vBNb
        J[6:9, 10] = dRnb_dtheta @ vBNb
        J[6:9, 11] = dRnb_dpsi @ vBNb

        # Jacobian for orientation update (df_Theta / domega)
        T = np.eye(3)
        T[0,1] = np.sin(phi) * np.tan(theta)
        T[0,2] = np.cos(phi) * np.tan(theta)
        T[1,1] = np.cos(phi)
        T[1,2] = -np.sin(phi)
        T[2,1] = np.sin(phi) / np.cos(theta)
        T[2,2] = np.cos(phi) / np.cos(theta)
        J[9:12, 3:6] = T

        # Jacobian for orientation update (df_Theta / dTheta)
        dT_dphi = np.array([
            [0, cphi*ttheta, -sphi*ttheta],
            [0, -sphi, -cphi],
            [0, cphi/ctheta, -sphi/ctheta]
        ])

        dT_dtheta = np.array([
            [0, sphi/ctheta**2, cphi/ctheta**2],
            [0, 0, 0],
            [0, sphi*stheta/ctheta**2, cphi*stheta/ctheta**2]
        ])

        J[9:12, 9] = dT_dphi @ omegaBNb
        J[9:12, 10] = dT_dtheta @ omegaBNb
        
        return f, J

    return f
