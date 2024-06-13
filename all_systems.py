import numpy as np

class System:
  def __init__(self, f1gen, f2gen, Hgen, spacedim, f1, f2, H, PE, KE, netshape1, netshape2, netshape3, sepshape1, sepshape2, sepshape3, sepshape4, x0, H0, h, LR):
    self.f1gen = f1gen 
    self.f2gen = f2gen 
    self.Hgen = Hgen 
    self.spacedim = spacedim
    self.f1 = f1
    self.f2 = f2
    self.H = H
    self.PE = PE 
    self.KE = KE
    self.netshape1 = netshape1
    self.netshape2 = netshape2
    self.netshape3 = netshape3 
    self.sepshape1 = sepshape1
    self.sepshape2 = sepshape2
    self.sepshape3 = sepshape3
    self.sepshape4 = sepshape4
    self.x0 = x0
    self.H0 = H0
    self.h = h
    self.LR = LR


pendulum = System(lambda x: x[1], lambda x: -np.sin(x[0]), lambda x: 1/2*x[1]**2+(1-np.cos(x[0])), [(-2*np.pi, 2*np.pi), (-1.2, 1.2)], 
			lambda z: z[:,1], lambda z: - np.sin(z[:,0]), lambda x: 1/2*x[:,1]**2+(1-np.cos(x[:,0])),
                        lambda x: (1-np.cos(x[:,0])), lambda x: 1/2*x[:,1]**2,
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001)

trigo = System(lambda z: - 2 * np.cos(z[1]) * np.sin(z[1]), lambda z: - 2 * np.cos(z[0]) * np.sin(z[0]), lambda z: np.sin(z[0])**2 + np.cos(z[1])**2, [(-2., 2.), (-2., 2.)], 
			lambda z: - 2 * np.cos(z[:,1]) * np.sin(z[:,1]), lambda z: - 2 * np.cos(z[:,0]) * np.sin(z[:,0]), lambda z: np.sin(z[:,0])**2 + np.cos(z[:,1])**2 -1,
                        lambda z: np.sin(z[:,0])**2, lambda z: np.cos(z[:,1])**2-1,
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001)

arctan = System(lambda z: 2 * z[1] / (z[1]**4 +1), lambda z: - 2 * z[0] / (z[0]**4 +1), lambda z: np.arctan(z[0]**2) + np.arctan(z[1]**2), [(-2., 2.), (-2., 2.)], 
			lambda z: 2 * z[:,1] / (z[:,1]**4 +1), lambda z: - 2 * z[:,0] / (z[:,0]**4 +1), lambda z: np.arctan(z[:,0]**2) + np.arctan(z[:,1]**2),
                        lambda z: np.arctan(z[:,0]**2), lambda z: np.arctan(z[:,1]**2),
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001)

logarithm = System(lambda x: - 1 + 3/x[1], lambda z: - 1 + 2 /z[0], lambda z: z[0] - np.log(z[0]**2) - z[1] + np.log(z[1]**3), [(1., 3.), (1., 3.)], 
			lambda x: - 1 + 3/x[:,1], lambda z: - 1 + 2 /z[:,0], lambda z: z[:,0] - np.log(z[:,0]**2) - z[:,1] + np.log(z[:,1]**3)+2,
                        lambda z:  z[:,0] - np.log(z[:,0]**2) -1, lambda z: - z[:,1] + np.log(z[:,1]**3) +1,
			2, 16, 1, 1, 11, 11, 1, 1., 2., 0.1, 0.001)

anisotropicoscillator2D = System(
			lambda x: np.asarray([x[2]/np.sqrt(x[2]**2+x[3]**2+1**2), x[3]/np.sqrt(x[2]**2+x[3]**2+1**2)]), 
			lambda x: -np.asarray([1*x[0]+0*x[0]**3,1*x[1]+0.05*x[1]**3]), 
			lambda x: np.sqrt(x[2]**2+x[3]**2+1**2) + 0.5*1*x[0]**2 +0.5*1*x[1]**2 +.25*0*x[0]**4 +.25*0.05*x[1]**4, 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,2]/np.sqrt(x[:,2]**2+x[:,3]**2+1**2), x[:,3]/np.sqrt(x[:,2]**2+x[:,3]**2+1**2)]), 
			lambda x: -np.stack([1*x[:,0]+0*x[:,0]**3,1*x[:,1]+0.05*x[:,1]**3]), 
			lambda x: np.sqrt(x[:,2]**2+x[:,3]**2+1**2) + 0.5*1*x[:,0]**2 +0.5*1*x[:,1]**2 +.25*0*x[:,0]**4 +.25*0.05*x[:,1]**4-1,
                        lambda x: 0.5*1*x[:,0]**2 +0.5*1*x[:,1]**2 +.25*0*x[:,0]**4 +.25*0.05*x[:,1]**4, 
                        lambda x: np.sqrt(x[:,2]**2+x[:,3]**2+1**2) -1,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01)

henonheiles = System(lambda x: np.asarray([x[2], x[3]]), 
			lambda x: np.asarray([-x[0]-2*1*x[0]*x[1], -x[1]-1*(x[0]*x[0]-x[1]*x[1])]), 
			lambda x: 1/2*(x[2]**2 + x[3]**2) +1/2*(x[0]**2+x[1]**2)+1*(x[0]**2 *x[1] -(x[1]**3)/3), 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,2], x[:,3]]), 
			lambda x: np.stack([-x[:,0]-2*1*x[:,0]*x[:,1], -x[:,1]-1*(x[:,0]*x[:,0]-x[:,1]*x[:,1])]), 
			lambda x: 1/2*(x[:,2]**2 + x[:,3]**2) +1/2*(x[:,0]**2+x[:,1]**2)+1*(x[:,0]**2 *x[:,1] -(x[:,1]**3)/3),
			lambda x: 1/2*(x[:,0]**2+x[:,1]**2)+1*(x[:,0]**2 *x[:,1] -(x[:,1]**3)/3),
			lambda x: 1/2*(x[:,2]**2 + x[:,3]**2),
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.01, 0.01)

todalattice = System(lambda x: np.asarray([x[3], x[4], x[5]]), 
			lambda x: np.asarray([-np.exp(x[0]-x[1])+np.exp(x[2]-x[0]),
                           -np.exp(x[1]-x[2])+np.exp(x[0]-x[1]),
                           -np.exp(x[2]-x[0])+np.exp(x[1]-x[2]),]), 
			lambda x: 0.5*(x[3]**2+x[4]**2+x[5]**2)+np.exp(x[0]-x[1])+np.exp(x[1]-x[2])+np.exp(x[2]-x[0])-3, 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,3], x[:,4], x[:,5]]), 
			lambda x: np.stack([-np.exp(x[:,0]-x[:,1])+np.exp(x[:,2]-x[:,0]),
                           -np.exp(x[:,1]-x[:,2])+np.exp(x[:,0]-x[:,1]),
                           -np.exp(x[:,2]-x[:,0])+np.exp(x[:,1]-x[:,2]),]), 
			lambda x: 0.5*(x[:,3]**2+x[:,4]**2+x[:,5]**2)+np.exp(x[:,0]-x[:,1])+np.exp(x[:,1]-x[:,2])+np.exp(x[:,2]-x[:,0])-3,
			lambda x: np.exp(x[:,0]-x[:,1])+np.exp(x[:,1]-x[:,2])+np.exp(x[:,2]-x[:,0])-3,
			lambda x: 0.5*(x[:,3]**2+x[:,4]**2+x[:,5]**2),
			6, 31, 1, 3, 22, 22, 1, 0., 0., 0.01, 0.01)

coupledoscillator = System(lambda x: np.asarray([x[3], x[4], x[5]]), 
			lambda x: np.asarray([1*(x[1]-x[0]), 1*(x[0]+x[2]) - 2*1*x[1], -1*(x[2]-x[1])]), 
			lambda x: 0.5*(x[3]**2 + x[4]**2 + x[5]**2) + 0.5* 1 *(x[1]-x[0])**2 + 0.5* 1 *(x[2]-x[1])**2 , 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,3],x[:,4],x[:,5]]), 
			lambda x: np.stack([1*(x[:,1]-x[:,0]), 1*(x[:,0]+x[:,2]) - 2*1*x[:,1], -1*(x[:,2]-x[:,1])]), 
			lambda x: 0.5*(x[:,3]**2 + x[:,4]**2 + x[:,5]**2) + 0.5* 1 *(x[:,1]-x[:,0])**2 + 0.5* 1 *(x[:,2]-x[:,1])**2,
			lambda x: 0.5* 1 *(x[:,1]-x[:,0])**2 + 0.5* 1 *(x[:,2]-x[:,1])**2,
			lambda x: 0.5*(x[:,3]**2 + x[:,4]**2 + x[:,5]**2), 
			6, 31, 1, 3, 22, 22, 1, 0., 0., 0.01, 0.01)

coupledoscillator10 = System(lambda x: np.asarray([x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19]]), 
			lambda x: np.asarray([1*(x[1]-x[0]), 1*(x[0]+x[2]) - 2*1*x[1], 
                           1*(x[1]+x[3]) - 2*1*x[2], 1*(x[2]+x[4]) - 2*1*x[3], 
                           1*(x[3]+x[5]) - 2*1*x[4], 1*(x[4]+x[6]) - 2*1*x[5],
                           1*(x[5]+x[7]) - 2*1*x[6], 1*(x[6]+x[8]) - 2*1*x[7],
                          1*(x[7]+x[9]) - 2*1*x[8], -1*(x[9]-x[8])]), 
			lambda x: 0.5*(x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2) + 0.5* 1 *(x[1]-x[0])**2 + 0.5* 1 *(x[2]-x[1])**2 + 0.5* 1 *(x[3]-x[2])**2 + 0.5* 1 *(x[4]-x[3])**2 + 0.5* 1 *(x[5]-x[4])**2 + 0.5* 1 *(x[6]-x[5])**2 + 0.5* 1 *(x[7]-x[6])**2 + 0.5* 1 *(x[8]-x[7])**2 + 0.5* 1 *(x[9]-x[8])**2, 
			[(-0.5,0.5)]*20, 
			lambda x: np.stack([x[:,10], x[:,11], x[:,12], x[:,13], x[:,14], x[:,15], x[:,16], x[:,17], x[:,18], x[:,19]]), 
			lambda x: np.stack([1*(x[:,1]-x[:,0]), 1*(x[:,0]+x[:,2]) - 2*1*x[:,1], 
                           1*(x[:,1]+x[:,3]) - 2*1*x[:,2], 1*(x[:,2]+x[:,4]) - 2*1*x[:,3], 
                           1*(x[:,3]+x[:,5]) - 2*1*x[:,4], 1*(x[:,4]+x[:,6]) - 2*1*x[:,5],
                           1*(x[:,5]+x[:,7]) - 2*1*x[:,6], 1*(x[:,6]+x[:,8]) - 2*1*x[:,7],
                          1*(x[:,7]+x[:,9]) - 2*1*x[:,8], -1*(x[:,9]-x[:,8])]), 
			lambda x: 0.5*(x[:,10]**2 + x[:,11]**2 + x[:,12]**2 + x[:,13]**2 + x[:,14]**2 + x[:,15]**2 + x[:,16]**2 + x[:,17]**2 + x[:,18]**2 + x[:,19]**2) + 0.5* 1 *(x[:,1]-x[:,0])**2 + 0.5* 1 *(x[:,2]-x[:,1])**2 + 0.5* 1 *(x[:,3]-x[:,2])**2 + 0.5* 1 *(x[:,4]-x[:,3])**2 + 0.5* 1 *(x[:,5]-x[:,4])**2 + 0.5* 1 *(x[:,6]-x[:,5])**2 + 0.5* 1 *(x[:,7]-x[:,6])**2 + 0.5* 1 *(x[:,8]-x[:,7])**2 + 0.5* 1 *(x[:,9]-x[:,8])**2,
			lambda x: 0.5* 1 *(x[:,1]-x[:,0])**2 + 0.5* 1 *(x[:,2]-x[:,1])**2 + 0.5* 1 *(x[:,3]-x[:,2])**2 + 0.5* 1 *(x[:,4]-x[:,3])**2 + 0.5* 1 *(x[:,5]-x[:,4])**2 + 0.5* 1 *(x[:,6]-x[:,5])**2 + 0.5* 1 *(x[:,7]-x[:,6])**2 + 0.5* 1 *(x[:,8]-x[:,7])**2 + 0.5* 1 *(x[:,9]-x[:,8])**2,
			lambda x: 0.5*(x[:,10]**2 + x[:,11]**2 + x[:,12]**2 + x[:,13]**2 + x[:,14]**2 + x[:,15]**2 + x[:,16]**2 + x[:,17]**2 + x[:,18]**2 + x[:,19]**2), 
			20, 44, 1, 10, 32, 32, 1, 0., 0., 0.01, 0.01)


print("imported systems")

