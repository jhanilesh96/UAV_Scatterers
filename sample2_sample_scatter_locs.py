import numpy as np
import scipy.constants as constants


func_x_max_ = lambda a, b, m, h_s_ : (b**2*m*h_s_ + a*b*(a**2 + b**2*m**2 - h_s_**2)**0.5) * (a**2 + b**2*m**2)**-1
func_x_min_ = lambda a, b, m, h_s_ : (b**2*m*h_s_ - a*b*(a**2 + b**2*m**2 - h_s_**2)**0.5) * (a**2 + b**2*m**2)**-1
func_a_int_ = lambda x_max, x_min, psi : (x_max - x_min) * (2 * np.cos(psi))**-1
func_x_off_ = lambda a_int, psi, x_min : a_int*np.cos(psi) + x_min
func_z_off_ = lambda m, x_off_, h_s_ : m*x_off_ - h_s_
func_b_int_ = lambda a,b,m,x_off_,h_s_ : (b**2 - ((b/a)**2)*((m*x_off_-h_s_)**2) - x_off_**2)**0.5
func_R_psi = lambda psi : np.array([[np.cos(psi),0,-np.sin(psi)],[0,1,0],[np.sin(psi),0,np.cos(psi)]])

# new is _ or x'-y'-z', old is x-y-z
func_new2old = lambda coords_, R, centre : coords_@R + centre
func_old2new = lambda coords, R_, centre : coords@R_ - centre

# utility
func_newz_from_newxoldz = lambda newx, oldz, psi, centre : (oldz - centre[2] + newx*np.sin(psi))/np.cos(psi)
func_newz_from_newxoldz = lambda newx, oldz, psi, centre : (oldz - centre[2] + newx*np.sin(psi))/np.cos(psi)

# '''' Sampled random variables '''
# h_s = np.random.rand()*h_max

# '''' Derived fixed variables '''
# tau = (2*f)/constants.speed_of_light
# a = f*(delta_tau/tau + 1)
# b = (a**2 - f**2)**0.5
# psi = (np.pi/2) - theta
# h_s_= f - (h_s/np.cos(psi))

# h = 2*f*np.sin(theta)
# r = 2*f*np.cos(theta)
# centre = x_c, y_c, z_c = r/2,0,h/2
# m = np.tan(psi)
# R = func_R_psi(psi);
# R_ = func_R_psi(-psi);

# x_max_ = func_x_max_(a,b,m,h_s_)
# x_min_ = func_x_min_(a,b,m,h_s_)
# a_int_ = func_a_int_(x_max_,x_min_,psi)
# x_off_ = func_x_off_(a_int_,psi,x_min_)
# z_off_ = func_z_off_(m,x_off_,h_s_)
# b_int_ = func_b_int_(a,b,m,x_off_,h_s_)
# a_int = a_int_
# b_int = b_int_
# z_max_ = func_newz_from_newxoldz(x_max_, h_s, psi, centre)
# z_min_ = func_newz_from_newxoldz(x_min_, h_s, psi, centre)

# coords_ = np.array([ [x_max_,0,z_max_],[x_min_,0,z_min_],[x_off_,0,z_off_],[x_off_,0,z_off_]])
# coords = func_new2old(coords_, R, centre)

# print(a,a_int_)
# print(b,b_int_,'\n')
# print(h_s,'\n')
# print(coords_,'\n')
# print(coords,'\n')
# exit()

# get the x_max, x_min for the ellipse (delta_tau) and h_s
def func_x_y_z(delta_tau, h_s, f, theta, return_only_fixed_params=False):
    '''' Derived fixed variables '''
    tau = (2*f)/constants.speed_of_light
    a = f*(delta_tau/tau + 1)
    b = (a**2 - f**2)**0.5
    psi = (np.pi/2) - theta
    h_s_= f - (h_s/np.cos(psi))

    h = 2*f*np.sin(theta)
    r = 2*f*np.cos(theta)
    centre = x_c, y_c, z_c = r/2,0,h/2
    m = np.tan(psi)
    R = func_R_psi(psi);
    R_ = func_R_psi(-psi);

    x_max_ = func_x_max_(a,b,m,h_s_)
    x_min_ = func_x_min_(a,b,m,h_s_)
    a_int_ = func_a_int_(x_max_,x_min_,psi)
    x_off_ = func_x_off_(a_int_,psi,x_min_)
    z_off_ = func_z_off_(m,x_off_,h_s_)
    b_int_ = func_b_int_(a,b,m,x_off_,h_s_)
    a_int = a_int_
    b_int = b_int_
    z_max_ = func_newz_from_newxoldz(x_max_, h_s, psi, centre)
    z_min_ = func_newz_from_newxoldz(x_min_, h_s, psi, centre)

    coords_ = np.array([ [x_max_,0,z_max_],[x_min_,0,z_min_],[x_off_,0,z_off_],[x_off_,0,z_off_]])
    coords = func_new2old(coords_, R, centre)
    
    x_max, x_min, x_off = coords[0][0], coords[1][0], coords[2][0]
    if return_only_fixed_params:
        return x_max, x_min, a_int, b_int
    x_s = np.random.uniform(low=x_min, high=x_max)
    y_x = (b_int/a_int)*((a_int)**2 - (x_s-x_off)**2)**0.5
    y_s = np.random.uniform(low=-y_x, high=y_x)
    print('delta_tau', delta_tau)
    print('a, a_int', a,a_int)
    print('b, b_int', b, b_int)
    print('x_max', x_max)
    print('x_min', x_min)
    print('xyz', x_s,y_s,h_s)
    print()
    return x_s, y_s, h_s, a_int, b_int

print('----------------------------------------------------')
print('----------------------------------------------------')
print('---------------------- PART 2 ----------------------')
print('----------------------------------------------------')
print('----------------------------------------------------')

''' '''

'''' base variables '''
f = 1500/2
theta = np.deg2rad(30)
delta_tau = 1.6e-6
# maximum excess delay
max_delta_tau = 1.6e-6              

h_min = 20
h_max = 200;


'''' derived random variables '''
# numper of delay taps
P = 5 + np.random.randint(5)
# delays for the taps
delta_taus = [(p+1)*max_delta_tau*(P)**-1 for p in range(P)]                
# parameter of number of scatterers for each delta tau/ellipsoid
lambda_Ps = [P*np.exp(-4*p*P**-1) for p in range(P)]



# number of scatteres for each ellipsoid
n_s = [np.random.poisson(lambda_Ps[i]) for i in range(P)]

# height of scatters
h_s = [np.random.uniform(low=h_min, high=h_max, size=n_s[p]) for p in range(P)]
# h_s = [np.zeros(shape=n_s[p]) for p in range(P)] # take for 0 hieght

# location of scatterers
loc_s_all = [[func_x_y_z(delta_taus[p], h_s[p][q], f, theta) for q in range(n_s[p])] for p in range(P)]
loc_s = [[loc_s_all[p][q][:3] for q in range(n_s[p])] for p in range(P)]

# 0 height ellipse intersection, ell_params : ellipse_params
ell_params = [func_x_y_z(delta_taus[p], 0, f, theta, return_only_fixed_params=True) for p in range(P)]




plot_ellipses = True;
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_ax = plt.gca()
_angs = np.linspace(0,2*np.pi,100)
for p in range(P):
    _x = (ell_params[p][0] + ell_params[p][1])/2
    _ = plt.plot(_x + ell_params[p][2]*np.cos(_angs), 0 + ell_params[p][3]*np.sin(_angs), color=plt.get_cmap('viridis')(p/P))
    for q in range(n_s[p]):
        _ = plt.plot(loc_s[p][q][0], loc_s[p][q][1], loc_s[p][q][2],color=plt.get_cmap('viridis')(p/P),marker='o')
        _ = plt.plot([loc_s[p][q][0],loc_s[p][q][0]], \
            [loc_s[p][q][1],loc_s[p][q][1]], \
                [0,loc_s[p][q][2]], \
                    color=plt.get_cmap('viridis')(p/P))

# BS
_ = plt.plot([0.0], [0.0], [0.0], color='red',marker='o')
# UAV
_ = plt.plot([2*f*np.cos(theta)],[0],[2*f*np.sin(theta)], color='red',marker='o')
_ = plt.plot([0,2*f*np.cos(theta)], [0,0], [0,2*f*np.sin(theta)], color='blue')
plt.show()

# print(0,'\t',int(ell_params[0][2]),'\t',int(ell_params[0][3]))
# for q in range(n_s[0]):
#     print(int(loc_s_all[0][q][2]), '\t', int(loc_s_all[0][q][3]), '\t', int(loc_s_all[0][q][4]))


