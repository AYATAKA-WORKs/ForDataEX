# 参考サイト : https://emotionexplorer.blog.fc2.com/blog-entry-345.html

import numpy as np
from scipy.optimize import leastsq, curve_fit, least_squares
import matplotlib.pyplot as plt

# Step1 : 最初にフィッテングしたい式の定義をする　-----------------------------------------------
def fitingCurve(prm,x) :
    phi_max, MUy_max, beta = prm[0], prm[1], prm[2]
    MUy = x
    return phi_max * (1 -  np.exp((MUy/MUy_max -1)*beta))

# Step2 : フィッテング式の値とデータサンプルとの誤差を計算する式を定義する　-----------------------
def objFunc(prm, x, y):
    phi_max, MUy_max, beta = prm[0], prm[1], prm[2]
    residual = y-fitingCurve(prm,x)
    return residual

# Step3 : データサンプルを用意　------------------------------------------------------------------
x = np.loadtxt('Muy_MU2vsMUy.csv',delimiter=',')
y = np.loadtxt('phi_MU2vsMUy.csv',delimiter=',')
xq  = np.linspace(np.amin(x),np.amax(x),1000) # プロット用に細かい定義域の変数を用意しておく

# Step3 : フィッテングするパラメータ (a,b) の探索前の初期値を設定　------------------------------
x0=np.array([0.1, 0.1, 0.1])

# Step4 : 非線形最小二乗計算 　---------------------------------------------------------------
# least_swuaresの解釈 x0がa,b,cにあたる独立変数, f_sacaleは許容誤差（デフォルト値は1e-8），argsがinputデータ
# 詳細URL:https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
# 以下では3種類の損失関数を使ってそれぞれ計算をしている

# 損失関数 linear
# result = least_squares(func, x0, loss='linear', f_scale=0.1, args=(x, y))
result = least_squares(objFunc, x0, loss='linear', args=(x, y))
phi_max, MUy_max, beta = result.x[0], result.x[1], result.x[2]
y_linear = fitingCurve(result.x, xq)
print("Linear : ")
print(phi_max, MUy_max, beta)

# 損失関数 soft_l1 
# result = least_squares(func, x0, loss='soft_l1', f_scale = 0.1, args=(x, y))
result = least_squares(objFunc, x0, loss='soft_l1', args=(x, y))
phi_max, MUy_max, beta = result.x[0], result.x[1], result.x[2]
y_soft_l1 = fitingCurve(result.x, xq)
print("Soft L1 : ")
print(phi_max, MUy_max, beta)

# 損失関数 cauchy 
# result = least_squares(func, x0, loss='cauchy', f_scale=0.1, args=(x, y))
result = least_squares(objFunc, x0, loss='cauchy', args=(x, y))
phi_max, MUy_max, beta = result.x[0], result.x[1], result.x[2]
y_cauchy = fitingCurve(result.x, xq)
print("Cauchy: ")
print(phi_max, MUy_max, beta)

# Step5 : グラフ表示　------------------------------------------------------------------------
plt.title('Test scipy.optimize.least_squares()')
plt.plot(x, y, 'bo', label='y-original')
plt.plot(xq, y_linear, color='red', label='y_linear')
plt.plot(xq, y_soft_l1, color='orange', label='y_soft_l1')
plt.plot(xq, y_cauchy, color='green', label='y_cauchy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('graph.png', dpi=350)
plt.show()