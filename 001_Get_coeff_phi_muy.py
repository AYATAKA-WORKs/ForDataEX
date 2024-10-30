# 参考サイト : https://emotionexplorer.blog.fc2.com/blog-entry-345.html

import numpy as np
from scipy.optimize import leastsq, curve_fit, least_squares
import matplotlib.pyplot as plt

# Step1 : フィッティングしたい式を定義 ------------------------------------------------------------

# phiとmuyの関係式を定義
def fitingCurve_phi(prm,x) :
    phi_max, MUy_max, beta = prm[0], prm[1], prm[2]
    MUy = x
    return phi_max * (1 -  np.exp((MUy/MUy_max -1)*beta))

# 多項式の定義式
def fitingCurve_poly(prm,x) :
    coeff = prm
    y = 0
    # y = coeff[0] * x**4 + coeff[1] * x**3 + coeff[2] * x**2 + coeff[3]* x + coeff[4]
    for i in range(len(coeff)) :
        y += coeff[i] * x ** (len(coeff)-(i+1))
    return y


# Step2 : フィッテング式の値とデータサンプルとの誤差を計算する式を定義する　--------------------------
def objFunc_phi(prm, x, y):
    residual = y-fitingCurve_phi(prm,x)
    return residual

def objFunc_poly(prm, x, y):
    residual = y-fitingCurve_poly(prm,x)
    return residual


# Step3 : データサンプルを用意　------------------------------------------------------------------
x_all = np.loadtxt('Muy_MU2vsMUy.csv',delimiter=',')
y_all = np.loadtxt('phi_MU2vsMUy.csv',delimiter=',')

# Step4 : phiとmuyの関係をフィッティング ---------------------------------------------------------
# フィッティングパラメータの初期化
phi_max = np.zeros(x_all.shape[0])
muy_max = np.zeros(x_all.shape[0])
beta = np.zeros(x_all.shape[0])
for i in range(x_all.shape[0]) :
    x = x_all[i]
    y = y_all[i]
    y = y[y>-0.05]
    x = x[0:len(y)]
    xq  = np.linspace(np.amin(x),np.amax(x),1000) # プロット用に細かい定義域の変数を用意しておく

    # フィッテングするパラメータ (a,b) の探索前の初期値を設定
    x0=np.array([0.1, 0.1, 0.1])

    # 非線形最小二乗計算
    # least_swuaresの解釈 x0がa,b,cにあたる独立変数, f_sacaleは許容誤差（デフォルト値は1e-8），argsがinputデータ
    # 詳細URL:https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    # 損失関数はlinear,soft_l1,cauchyから選ぶ
    # result = least_squares(func, x0, loss='linear', f_scale=0.1, args=(x, y))
    result = least_squares(objFunc_phi, x0, loss='linear', args=(x, y))
    phi_max[i], muy_max[i], beta[i] = result.x[0], result.x[1], result.x[2]
    yq = fitingCurve_phi(result.x, xq)
    #print("Linear : ")
    #print(phi_max, muy_max, beta)

    # グラフ作成　------------------------------------------------------------------------
    # plt.title('Test scipy.optimize.least_squares()')
    plt.plot(yq,xq, color='red', label='y_linear')
    plt.plot(y, x, 'bx', label='y-original')
    plt.xlabel('x')
    plt.ylabel('y')

np.savetxt('phi_max.csv', phi_max)
np.savetxt('muy_max.csv', muy_max)
np.savetxt('beta.csv', beta)

plt.xlim(0,np.amax(y_all))
plt.ylim(0,0.25)
plt.grid(True)
plt.show()

# Step5 : MU2との関係をフィッティング ---------------------------------------------------------
MU2_sample = np.loadtxt('MU2.csv',delimiter=',')
# フィッティングパラメータの初期化
coeff_a = np.ones(5) * 0.1
coeff_b = np.ones(3) * 0.1
coeff_c = np.ones(6) * 0.1
MU2q  = np.linspace(np.amin(MU2_sample),np.amax(MU2_sample),1000) # プロット用に細かい定義域の変数を用意しておく

# Figure用意

# 非線形最小二乗計算
# coeff a --- phi_max
result = least_squares(objFunc_poly, coeff_a, loss='soft_l1', args=(MU2_sample, phi_max))
coeff_a = result.x
phi_maxq = fitingCurve_poly(result.x, MU2q)
# coeff b --- beta
result = least_squares(objFunc_poly, coeff_b, loss='linear', args=(MU2_sample, beta))
coeff_b = result.x
betaq = fitingCurve_poly(result.x, MU2q)
# coeff_c --- muy_max
result = least_squares(objFunc_poly, coeff_c, loss='linear', args=(MU2_sample, muy_max))
coeff_c = result.x
muy_maxq = fitingCurve_poly(result.x, MU2q)

np.savetxt('coeff_a.csv', coeff_a)
np.savetxt('coeff_b.csv', coeff_b)
np.savetxt('coeff_c.csv', coeff_c)

fig = plt.figure()
plt.subplot(3,1,1)
plt.plot(MU2q,phi_maxq, color='red', label='y_linear')
plt.plot(MU2_sample, phi_max, 'bo', label='y-original')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(MU2q,betaq, color='red', label='y_linear')
plt.plot(MU2_sample, beta, 'bo', label='y-original')
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(MU2q,muy_maxq, color='red', label='y_linear')
plt.plot(MU2_sample, muy_max, 'bo', label='y-original')
plt.grid(True)
plt.show()
