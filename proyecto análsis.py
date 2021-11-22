import tkinter
from tkinter import *
from tkinter.ttk import *
import sympy as sp
import numpy as np
import sys

mainW= tkinter.Tk()
mainW.title("Proyecto Ánalisis Numérico")
mainW.geometry("300x300")
mainW.config(bg="gray87")

def gaussJordan(n,m):
  n=int(n)
  Slip=m.split(';')
  for i in range(len(Slip)): 
      Slip[i]=Slip[i].split(',')
  m=[[int(x) for x in group]for group in Slip]

  x = np.zeros(n)
  for i in range(n):
    if m[i][i] == 0.0:
        sys.exit('¡División por 0!')
        
    for j in range(n):
        if i != j:
            ratio = m[j][i]/m[i][i]

            for k in range(n+1):
                m[j][k] = m[j][k] - ratio * m[i][k]
  for i in range(n): 
      x[i] = m[i][n]/m[i][i]
  Str='\nLa solución es: '
  sol=''
  for i in range(n):
    sol= sol+'X%d = %0.2f   ' %(i+1,x[i]) + '\n'
  print(sol)
  windowResul=Toplevel(mainW)
  windowResul.title("Solución")
  windowResul.geometry("200x300")
  windowResul.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul, text=Str, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.25, anchor = CENTER)
  label2 = tkinter.Label( windowResul, text=sol, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.50, anchor = CENTER)

def PivParcial(m):
  #Adapción de formato
  A = m.split(';')
  n = len(A)
  for i in range(n):
    A[i] = A[i].split(',')
    for j in range(len(A[i])):
      A[i][j] = float(A[i][j])

  a = np.zeros((n,n))
  for i in range(n):
    a[i] = A[i][:-1]
    if all([f == 0 for f in A[i]]):
      sys.exit('Hay infinitas soluciones')
    else:
      if all([e == 0 for e in a[i]]):
        sys.exit('No hay solucion')
      else:
        continue  

  #Cambiar filas
  for k in range(n):
   for i in range(k,n):
     if abs(A[i][k]) > abs(A[k][k]):
        A[k], A[i] = A[i],A[k]
     else:
        continue

   #Hacer matriz triangular superior
   for j in range(k+1,n):
       ratio = float(A[j][k]) / A[k][k]
       for m in range(k, n+1):
          A[j][m] -=  ratio * A[k][m]
  
  #Se encuentra la última x
  x = [0 for i in range(n)]
  x[n-1] =float(A[n-1][n])/A[n-1][n-1]

  Str='\nLa solución es: '
  sol=''
  #Sustitución regresiva
  for i in range (n-1,-1,-1):
    z = 0
    for j in range(i+1,n):
        z = z  + float(A[i][j])*x[j]
    x[i] = float(A[i][n] - z)/A[i][i]
    sol = sol + 'X%d = %0.2f   ' %(i+1,x[i]) +'\n'
  print(sol)  
  windowResul2=Toplevel(mainW)
  windowResul2.title("Solución")
  windowResul2.geometry("200x300")
  windowResul2.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul2, text=Str, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.25, anchor = CENTER)
  label2 = tkinter.Label( windowResul2, text=sol, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.50, anchor = CENTER)

def PivTotal(mat):

  #Adapción de formato
  A = mat.split(';')
  n = len(A)
  for i in range(n):
    A[i] = A[i].split(',')
    for j in range(len(A[i])):
      A[i][j] = float(A[i][j])
  
  #Nombres de variables:
  nom = []
  for i in range(n):
    nom.append('X%d' %(i+1))
  
  A = np.array(A)
  a = np.zeros((n,n))
  for i in range(n):
    a[i] = A[i][:-1]
    if all([f == 0 for f in A[i]]):
      sys.exit('Hay infinitas soluciones')
    else:
      if all([e == 0 for e in a[i]]):
        sys.exit('No hay solucion')
      else:
        continue  

  #Máximo
  for k in range(n):
    maxi = []
    for i in range(k,n):
      maxi.append(max(abs(A[i,:-1])))
      maxx = max(maxi)
      indice = np.where(abs(A)==maxx)
 
    #Cambio de columnas
    temp = nom[k]
    nom[k] = nom[indice[1][0]]
    nom[indice[1][0]] = temp

    Columntemp = A[:,k].copy()
    A[:,k] = A[:,indice[1][0]]
    A[:,indice[1][0]] = Columntemp
    
    
    #Cambio de filas
    Filatemp = A[k,:].copy()
    A[k,:] = A[indice[0][0],:]
    A[indice[0][0],:] = Filatemp
    
    #Hacer matriz triangular superior
    for j in range(k+1,n):
        ratio = float(A[j][k]) / A[k][k]
        for m in range(k, n+1):
            A[j][m] -=  ratio * A[k][m]
    
  #Se encuentra la última x
  x = [0 for i in range(n)]
  x[n-1] =float(A[n-1][n])/A[n-1][n-1]
  Str = '\nLa solución es: '
  sol = ''
  #Sustitución regresiva
  for i in range (n-1,-1,-1):
    z = 0
    for j in range(i+1,n):
        z = z  + float(A[i][j])*x[j]
    x[i] = float(A[i][n] - z)/A[i][i]
    sol = sol + str(nom[i]) + '=' + str((x[i])) + '\n'
  print(sol)
  windowResul3=Toplevel(mainW)
  windowResul3.title("Solución")
  windowResul3.geometry("200x300")
  windowResul3.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul3, text=Str, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.25, anchor = CENTER)
  label2 = tkinter.Label( windowResul3, text=sol, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.50,anchor = CENTER)

def LU(mat,b1):
  A = mat.split(';')
  n = len(A)
  for i in range(n):
    A[i] = A[i].split(',')
    for j in range(len(A[i])):
      A[i][j] = float(A[i][j])
  A = np.array(A)
  b = b1.split(',')
  for i in range(len(b)):
    b[i] = float(b[i])

  sizeA = A.shape[0]
  P = np.identity(sizeA)
  L = P
  U = A
  PF = P
  LF = np.zeros((sizeA,sizeA))

  for i in range (0,sizeA-1):
    indice = np.argmax(abs(U[i:,i]))
    indice = indice + i
    if indice != i:
      P[[indice,i],i:sizeA] = P[[i,indice],i:sizeA]
      print(P[[indice,i],i:sizeA])
      U[[indice,i],i:sizeA] = P[[i,indice],i:sizeA]
      PF = np.dot(P,PF)
      LF = np.dot(P,LF)
    L = np.identity(sizeA)
    for j in range(i+1,sizeA):
      if U[i,i] == 0:
        sys.exit('¡División por cero!')
      else:
        LF[j,i] = (U[j,i]/U[i,i])
        L[j,i] = -(U[j,i]/U[i,i])
    U = np.dot(L,U)
    np.fill_diagonal(LF,1)
    STR = 'La factorización de la matriz es : ' 
    L= str(LF) 
    U= str(U)
  windowResul3=Toplevel(mainW)
  windowResul3.title("Solución")
  windowResul3.geometry("300x300")
  windowResul3.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul3, text=STR, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.25, anchor = CENTER)
  label2 = tkinter.Label( windowResul3, text='L = ' + L, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.35, anchor = CENTER)
  label3 = tkinter.Label( windowResul3, text='U = ' +U, bg ="LavenderBlush", fg = "black")
  label3.pack()
  label3.place(relx=0.5,rely=0.7, anchor = CENTER)

def jacobi(mat,b1,n):
  A = mat.split(';')
  s = len(A)
  n = int(n)
  for i in range(s):
    A[i] = A[i].split(',')
    for j in range(len(A[i])):
      A[i][j] = float(A[i][j])
  b = b1.split(',')
  for i in range(len(b)):
    b[i] = float(b[i])

  salida = ''                                                                                                                                                          

  x = np.zeros(len(A[0]))                                                                                                                                                   
  Ddiag = np.diag(A)
  R = A - np.diagflat(Ddiag)
                                                                                                                                                                     
  for i in range(n):
    x = (b - np.dot(R,x)) / Ddiag
  
  L = - np.tril(A,-1)
  U = - np.triu(A,1)
  D = A+L+U
  Tj = np.matmul(np.linalg.inv(D),(L+U))
  lambdas, autoVec = np.linalg.eig(Tj)
 
  maxLambda = np.amax(abs(lambdas))

  if maxLambda >= 1:
    salida = "No tiene solución"

  salida = ('El método converge en ' + '\n' + str(x))

  print(salida)
  windowResul2=Toplevel(mainW)
  windowResul2.title("Solución")
  windowResul2.geometry("200x300")
  windowResul2.config(bg="LavenderBlush")
  label2 = tkinter.Label( windowResul2, text=salida, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.50, anchor = CENTER)

def gaussSeidel(A,x,b):
  s = len(A)             
   
  for j in range(0, s):
    d = b[j]
    for i in range(0, s):
      if(j != i):
        d-=A[j][i] * x[i]       
        x[j] = d / A[j][j]         

  return x

def seidel(mat,b1, numIt):
  A = mat.split(';')
  s = len(A)
  for i in range(s):
    A[i] = A[i].split(',')
    for j in range(len(A[i])):
      A[i][j] = float(A[i][j])

  b = b1.split(',')
  for i in range(len(b)):
    b[i] = float(b[i]) 
  x = np.zeros(len(b))

  L = - np.tril(A,-1)
  U = - np.triu(A,1)
  D = A+L+U
  Tj = np.matmul(np.linalg.inv(D),(L+U))
  sizeA = np.size(A,0)

  C = np.matmul(np.linalg.inv(D),b)
    
  lambdas, autoVec = np.linalg.eig(Tj)
  maxLambda = np.amax(abs(lambdas))

  if maxLambda >= 1:
    salida = ("No tiene solución") 

  numIt = int(numIt)
  for i in range(0, numIt):
    x = gaussSeidel(A,x,b) 

  salida =  ('El método converge en ' + '\n' + str(x))       
    
  print(salida)
  windowResul2=Toplevel(mainW)
  windowResul2.title("Solución")
  windowResul2.geometry("200x300")
  windowResul2.config(bg="LavenderBlush")
  label2 = tkinter.Label( windowResul2, text=salida, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.50, anchor = CENTER)


def lagrange(x,y,xp):
  x=x.split(',')
  x=[float(xs) for xs in x]
  y=y.split(',')
  y=[float(ys) for ys in y]
  yp = 0
  n=len(x)
  xp=float(xp)
  if len(x) != len(y):
        sys.exit('Ingrese coordenadas validas')
  setx=set(x)
  if len(x) != len(setx):
      sys.exit('Hay imagenes con la misma preimagen')
      

  for i in range(n):
    
    p = 1
    
    for j in range(n):
        if i != j:
            p = p * (xp - x[j])/(x[i] - x[j])
    
    yp = yp + p * y[i] 
   
  if xp>=max(x) or xp<=min(x):
      Str="Punto fuera de lo límites"
  else:
    Str="Interpolated value at %.3f is %.3f." % (xp, yp)
    Str=str(Str)

  windowResul=Toplevel(mainW)
  windowResul.title("Solución")
  windowResul.geometry("200x200")
  windowResul.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul, text=Str, bg = "LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5,anchor=CENTER)

def quadSplines(x,y,pto):

  #Adapción de formato
  x = x.split(',')
  n = len(x)
  for i in range(n):
    x[i] = float(x[i])
  pto = float(pto)
    
  y = y.split(',')
  m = len(y)
  for i in range(n):
    y[i] = float(y[i])

  #Revisar que haya el mismo número de x e y
  if n<m:
    sys.exit('ERROR: No todas las imagenes tienen preimagen')
  if m<n:
    sys.exit('ERROR: No todas las preimagenes tienen imagen')
  #Revisar si hay puntos coordenados con la misma x
  vistos = set()
  unicos = []
  for i in x:
    if i not in vistos:
      unicos.append(i)
      vistos.add(i)
  if unicos != x:
    sys.exit('ERROR: Hay imágenes con la misma preimagen')

  Y = np.array(y)
  X = np.array(x)

  if pto > max(X) or pto < min(X):
    sys.exit('El punto a interpolar está afuera de los límites')

  y = [(Y[i//2]if i%2==0 else Y[(i)//2]) if i <= 2*(n-1) else 0 for i in range(1, 3*(n-1) + 1)]
  
  A = np.zeros((3*(n-1),3*(n-1)))

  for i in range(n-1):
    A[2*(i + 1) - 2][3*i] = 1
    A[2*(i + 1) - 1][3*i] = 1 
    A[2*(i + 1) - 2][3*i + 1] = X[i]
    A[2*(i + 1) - 2][3*i + 2] = X[i]**2
    A[2*(i + 1) - 1][3*i + 1] = X[i + 1]
    A[2*(i + 1) - 1][3*i + 2] = X[i + 1]**2

  for i in range(n-2):
    A[2*(n-1) + i][3*i + 1] = 1
    A[2*(n-1) + i][3*i + 4] = -1 
    A[2*(n-1) + i][3*i + 2] = 2*X[i + 1]
    A[2*(n-1) + i][3*i + 5] = -2*X[i + 1]

  A[3*(n-1) - 1][2] = 2

  A = np.linalg.inv(A)
  coef = np.matmul(A,y)
  p=[]
  cont = 0
  x = sp.symbols('x') 
  for j in range(n-1):
    pj = coef[cont] + coef[cont+1]*x + coef[cont+2]*x**2
    cont=cont+3
    p.append(pj)
  
  for i in range(1,n):
    if pto>=X[i-1] and pto<=X[i]:
      sol = p[i-1].subs(x,pto)
    else:
      continue
  print(sol)
  windowResul3=Toplevel(mainW)
  windowResul3.title("Solución")
  windowResul3.geometry("350x200")
  windowResul3.config(bg="LavenderBlush")
  label2 = tkinter.Label( windowResul3, text=sol, bg ="LavenderBlush", fg = "black")
  label2.pack()
  label2.place(relx=0.5,rely=0.50,anchor = CENTER)

def coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

    return a

def newton_polynomial(x, y, xp):
    x=x.split(',')
    x=[float(xs) for xs in x]
    y=y.split(',')
    y=[float(ys) for ys in y]
    xp=float(xp)
    setx = set(x)
    if len(x) != len(y):
        sys.exit('Ingrese coordenadas validas')
    if len(x) != len(setx):
      sys.exit('Hay imagenes con la misma preimagen')

    a = coefficient(x, y)
    n = len(x) - 1  
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (xp - x[n - k])*p

    if xp>=max(x) or xp<=min(x):
        Str="Punto fuera de los límites"
    else:
        Str="Interpolated value at %.3f is %.3f." % (xp, p)
        Str=str(Str)
        
    windowResul=Toplevel(mainW)
    windowResul.title("Solución")
    windowResul.geometry("200x200")
    windowResul.config(bg="LavenderBlush")
    label = tkinter.Label( windowResul, text=Str, bg ="LavenderBlush", fg = "black")
    label.pack()
    label.place(relx=0.5,rely=0.5,anchor=CENTER)

def busquedas(fx,x0,paso,ni): 
  fx = sp.sympify(fx)
  x0 = float(x0)
  paso = float(paso)
  ni = float(ni)
  x = sp.Symbol('x')
  salida = ''
  if fx.subs(x,x0) == 0:
    return str(x0) + ' es raíz'
  else:
    xn = x0 + paso
    i = 0
    while (ni>i and float(fx.subs(x,x0))*float(fx.subs(x,xn))>0):
      x0 = xn
      xn = xn + paso
      i += 1
  if fx.subs(x,xn) == 0:
    salida = str(xn) + ' es raíz'
  elif float(fx.subs(x,x0))*float(fx.subs(x,xn))<0:
    salida = 'entre ' + str((x0,xn)) + ' hay raíz'
  else:
    salida = 'no se encontró raíz'
  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)
  windowResul4.update()

def bisecciones(fx,xi,xf,n,t):
  fx = sp.sympify(fx)
  x = sp.Symbol('x')
  xf = float(xf)
  xi = float(xi)
  n = float(n)
  t = float(t)

  salida = ''
  if float(fx.subs(x,xi))*float(fx.subs(x,xf)) == 0:
    salida = str(xi) + 'O' + str(xf) + 'son raíces'
  elif float(fx.subs(x,xi))*float(fx.subs(x,xf)) > 0:
    salida = 'no hay raíz'
  else:
    xm = (xi + xf)/2
    cont = 0
    error = abs(xi-xm)
    while(error > t and n > cont and float(fx.subs(x,xm))!= 0):
      if float(fx.subs(x,xi))*float(fx.subs(x,xm)) < 0:
        xf = xm 
      else:
        xi = xm 
      xm = (xi+xf)/2
      error = abs(xm-xi)
      cont += 1
    if fx.subs(x,xm) == 0:
      salida = str(xm)+ ' es raíz'
    elif error < t:
      salida = str(xm) + ' es raíz con tolerancia ' + str(t)
    else:
      salida = 'no se encuentra solución'
  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)

def puntoFijo(f,x0,n, t):
  f = sp.sympify(f)
  x = sp.Symbol('x')
  x0 = float(x0)
  n = float(n)
  t = float(t)
  g=-(f-x)
  i = 0

  salida = ''
  error = t + 1
  while (i < n and error > t):
    xn = float(g.subs(x,x0))
    error = abs(xn - x0)
    i += 1
    x0 = xn
  if error <= t:
    salida = str(xn) + ' es raíz con tolerancia ' + str(t)
  else:
    salida = 'No converge'
  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)

def reglaFalsa (xi, xf, f, t, n):
  f = sp.sympify(f)
  x = sp.Symbol('x')
  xf = float(xf)
  xi = float(xi)
  n = float(n)
  t = float(t)

  salida = ''

  if (f.subs(x,xi) * f.subs(x,xf) == 0):
    if (f.subs(x,xi) == 0):
      salida =("Hay una raíz de en " + str(xi))
    if (f.subs(x,xf) == 0):
      salida = ("Hay una raíz en " + str(xf))
  elif (f.subs(x,xi) * f.subs(x,xf) > 0):
    salida = ("No se encuentra solución")
  else: 
    xm = xf -((f.subs(x,xf)*(xi-xf))/(f.subs(x,xi)-f.subs(x,xf)))
    xm = float(xm)
    ni = 0
    error = abs(xi-xm)
    while (error>t and ni<n and f.subs(x,xm) != 0):
      if (f.subs(x,xi) * f.subs(x,xm) < 0):
        xf = xm
      else: 
        xi = xm
      xm = xf -((f.subs(x,xf)*(xi-xf))/(f.subs(x,xi)-f.subs(x,xf)))
      xm = float(xm)
      error = abs(xm - xf)
      ni += 1
    if (f.subs(x,xm) == 0):
      salida = ("Se halló una raíz en " + str(xm) )
    elif error < t: 
      salida = ( str(xm) + " Es raíz con una toleracia  " + str(t))
    else:
      salida = ("No se encuentra una solución")

  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)


def newton(fx,x0,n,t):
  fx = sp.sympify(fx)
  x = sp.Symbol('x')
  x0 = float(x0)
  n = float(n)
  t = float(t)
  dfx = sp.diff(fx,x)
  cont = 0

  salida = ''
  error = t + 1
  while(cont < n and error > t):
    xn = x0 - float(fx.subs(x,x0))/float(dfx.subs(x,x0))
    error = abs(xn-x0)
    cont += 1
    x0 = xn
  if error <= t:
    salida = str(xn) + ' es raíz con tolerancia: ' + str(t)
  else:
    salida = 'No converge'
  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)

def secante(fx,x0,xn,n,t):
  fx = sp.sympify(fx)
  x = sp.Symbol('x')
  x0 = float(x0)
  xn = float(xn)
  n = float(n)
  t = float(t)

  salida = ''

  cont = 0
  error = t + 1
  v = 1
  while(cont < n and error > t):
    if float(fx.subs(x,xn))-float(fx.subs(x,x0)) ==  0: 
      salida =  ('¡Divisón por 0! Por  favor eliga otros x0,xn')
      v = 0
      break
    xn1 = xn-((((xn-x0)/(float(fx.subs(x,xn))-float(fx.subs(x,x0)))))*float(fx.subs(x,xn)))
    x0 = xn
    xn = xn1
    error = abs(xn-x0)
    cont += 1
  if error <= t:
    salida = str(xn) + ' es raíz con tolerancia: ' + str(t)
  elif v == 1:
    salida = 'No converge'

  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)

def raices(fx,x0,n,t):
  fx = sp.sympify(fx)
  x = sp.Symbol('x')
  x0 = float(x0)
  n = float(n)
  t = float(t)

  salida = ''

  dfx = sp.diff(fx,x)
  fx= fx/dfx
  dfx = sp.diff(fx,x)
  cont = 0
  error = t + 1
  while(cont < n and error > t):
    xn = x0 - float(fx.subs(x,x0))/float(dfx.subs(x,x0))
    error = abs(xn-x0)
    cont += 1
    x0 = xn
  if error <= t:
    salida = str(xn) + ' es raíz con tolerancia: ' + str(t)
  else:
    salida = 'No converge'

  print(salida)
  windowResul4=Toplevel(mainW)
  windowResul4.title("Solución")
  windowResul4.geometry("350x200")
  windowResul4.config(bg="LavenderBlush")
  label = tkinter.Label( windowResul4, text=salida, bg ="LavenderBlush", fg = "black")
  label.pack()
  label.place(relx=0.5,rely=0.5, anchor = CENTER)


def openwindowBusquedas():
    windowBusquedas=Toplevel(mainW)
    windowBusquedas.title("Búsquedas")
    windowBusquedas.geometry("200x200")
    windowBusquedas.config(bg="Lavender")
    e1 = tkinter.Label(windowBusquedas, text = "x inicial o x0", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.50, anchor=CENTER)
    x0=tkinter.Entry(windowBusquedas)
    x0.pack()
    x0.place(relx=0.2,rely=0.60, anchor=CENTER, width = 40)
    e1 = tkinter.Label(windowBusquedas, text = "Paso", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.20, anchor=CENTER)
    paso=tkinter.Entry(windowBusquedas)
    paso.pack()
    paso.place(relx=0.2,rely=0.30, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowBusquedas, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.8,rely=0.50, anchor=CENTER)
    ni=tkinter.Entry(windowBusquedas)
    ni.pack()
    ni.place(relx=0.8,rely=0.60, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowBusquedas, text = "Función", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.8,rely=0.20, anchor=CENTER)
    fx=tkinter.Entry(windowBusquedas)
    fx.pack()
    fx.place(relx=0.8,rely=0.30, anchor=CENTER, width = 40)
    botonCal=tkinter.Button(windowBusquedas,text="Calcular", bg="Gainsboro",command=lambda:busquedas(fx.get(),x0.get(),paso.get(),ni.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.80, anchor=CENTER) 

def openwindowBisecciones():
    windowBisecciones=Toplevel(mainW)
    windowBisecciones.title("Bisecciones")
    windowBisecciones.geometry("200x200")
    windowBisecciones.config(bg="Lavender")
    e1 = tkinter.Label(windowBisecciones, text = "x inicial", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.3,rely=0.10, anchor=CENTER)
    xi=tkinter.Entry(windowBisecciones)
    xi.pack()
    xi.place(relx=0.30,rely=0.20, anchor=CENTER, width = 40)
    e2 = tkinter.Label(windowBisecciones, text = "x final", bg = "Lavender", fg = "black")
    e2.pack()
    e2.place(relx=0.3,rely=0.30, anchor=CENTER)
    xf=tkinter.Entry(windowBisecciones)
    xf.pack()
    xf.place(relx=0.30,rely=0.40, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowBisecciones, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.7,rely=0.10, anchor=CENTER)
    n=tkinter.Entry(windowBisecciones)
    n.pack()
    n.place(relx=0.70,rely=0.20, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowBisecciones, text = "Tolerancia", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.7,rely=0.30, anchor=CENTER)
    t=tkinter.Entry(windowBisecciones)
    t.pack()
    t.place(relx=0.70,rely=0.40, anchor=CENTER, width = 40)
    e5 = tkinter.Label(windowBisecciones, text = "Función", bg = "Lavender", fg = "black")
    e5.pack()
    e5.place(relx=0.5,rely=0.50, anchor=CENTER)
    fx=tkinter.Entry(windowBisecciones)
    fx.pack()
    fx.place(relx=0.5,rely=0.60, anchor=CENTER)
    botonCal=tkinter.Button(windowBisecciones,text="Calcular", bg="Gainsboro",command=lambda:bisecciones(fx.get(),xf.get(),xi.get(),n.get(),t.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.80, anchor=CENTER) 

def openwindowPuntoF():
    windowPunto=Toplevel(mainW)
    windowPunto.title("Punto Fijo")
    windowPunto.geometry("200x200")
    windowPunto.config(bg="Lavender")
    e1 = tkinter.Label(windowPunto, text = "x inicial o x0", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.50, anchor=CENTER)
    x0=tkinter.Entry(windowPunto)
    x0.pack()
    x0.place(relx=0.2,rely=0.60, anchor=CENTER, width = 40)
    e1 = tkinter.Label(windowPunto, text = "Tolerancia", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.20, anchor=CENTER)
    t=tkinter.Entry(windowPunto)
    t.pack()
    t.place(relx=0.2,rely=0.30, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowPunto, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.8,rely=0.50, anchor=CENTER)
    n=tkinter.Entry(windowPunto)
    n.pack()
    n.place(relx=0.8,rely=0.60, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowPunto, text = "Función", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.8,rely=0.20, anchor=CENTER)
    fx=tkinter.Entry(windowPunto)
    fx.pack()
    fx.place(relx=0.8,rely=0.30, anchor=CENTER, width = 40)
    botonCal=tkinter.Button(windowPunto,text="Calcular", bg="Gainsboro",command=lambda:puntoFijo(fx.get(),x0.get(),n.get(),t.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.90, anchor=CENTER) 

def openwindowReglaf():
    windowregla=Toplevel(mainW)
    windowregla.title("Regla Falsa")
    windowregla.geometry("200x200")
    windowregla.config(bg="Lavender")
    e1 = tkinter.Label(windowregla, text = "x inicial", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.3,rely=0.10, anchor=CENTER)
    xi=tkinter.Entry(windowregla)
    xi.pack()
    xi.place(relx=0.30,rely=0.20, anchor=CENTER, width = 40)
    e2 = tkinter.Label(windowregla, text = "x final", bg = "Lavender", fg = "black")
    e2.pack()
    e2.place(relx=0.3,rely=0.30, anchor=CENTER)
    xf=tkinter.Entry(windowregla)
    xf.pack()
    xf.place(relx=0.30,rely=0.40, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowregla, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.7,rely=0.10, anchor=CENTER)
    n=tkinter.Entry(windowregla)
    n.pack()
    n.place(relx=0.70,rely=0.20, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowregla, text = "Tolerancia", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.7,rely=0.30, anchor=CENTER)
    t=tkinter.Entry(windowregla)
    t.pack()
    t.place(relx=0.70,rely=0.40, anchor=CENTER, width = 40)
    e5 = tkinter.Label(windowregla, text = "Función", bg = "Lavender", fg = "black")
    e5.pack()
    e5.place(relx=0.5,rely=0.50, anchor=CENTER)
    fx=tkinter.Entry(windowregla)
    fx.pack()
    fx.place(relx=0.5,rely=0.60, anchor=CENTER)
    botonCal=tkinter.Button(windowregla,text="Calcular", bg="Gainsboro",command=lambda:reglaFalsa(xi.get(),xf.get(),fx.get(),t.get(),n.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.80, anchor=CENTER) 

def openwindowNewton():
    windowNewton=Toplevel(mainW)
    windowNewton.title("Newton")
    windowNewton.geometry("200x200")
    windowNewton.config(bg="Lavender")
    e1 = tkinter.Label(windowNewton, text = "x inicial o x0", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.50, anchor=CENTER)
    x0=tkinter.Entry(windowNewton)
    x0.pack()
    x0.place(relx=0.2,rely=0.60, anchor=CENTER, width = 40)
    e1 = tkinter.Label(windowNewton, text = "Tolerancia", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.20, anchor=CENTER)
    t=tkinter.Entry(windowNewton)
    t.pack()
    t.place(relx=0.2,rely=0.30, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowNewton, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.8,rely=0.50, anchor=CENTER)
    n=tkinter.Entry(windowNewton)
    n.pack()
    n.place(relx=0.8,rely=0.60, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowNewton, text = "Función", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.8,rely=0.20, anchor=CENTER)
    fx=tkinter.Entry(windowNewton)
    fx.pack()
    fx.place(relx=0.8,rely=0.30, anchor=CENTER, width = 40)
    botonCal=tkinter.Button(windowNewton,text="Calcular", bg="Gainsboro",command=lambda:newton(fx.get(),x0.get(),n.get(),t.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.90, anchor=CENTER) 
    

def openwindowSecante():
    windowsecante=Toplevel(mainW)
    windowsecante.title("Secante")
    windowsecante.geometry("200x200")
    windowsecante.config(bg="Lavender")
    e1 = tkinter.Label(windowsecante, text = "x inicial", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.3,rely=0.10, anchor=CENTER)
    xi=tkinter.Entry(windowsecante)
    xi.pack()
    xi.place(relx=0.30,rely=0.20, anchor=CENTER, width = 40)
    e2 = tkinter.Label(windowsecante, text = "x final", bg = "Lavender", fg = "black")
    e2.pack()
    e2.place(relx=0.3,rely=0.30, anchor=CENTER)
    xf=tkinter.Entry(windowsecante)
    xf.pack()
    xf.place(relx=0.30,rely=0.40, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowsecante, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.7,rely=0.10, anchor=CENTER)
    n=tkinter.Entry(windowsecante)
    n.pack()
    n.place(relx=0.70,rely=0.20, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowsecante, text = "Tolerancia", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.7,rely=0.30, anchor=CENTER)
    t=tkinter.Entry(windowsecante)
    t.pack()
    t.place(relx=0.70,rely=0.40, anchor=CENTER, width = 40)
    e5 = tkinter.Label(windowsecante, text = "Función", bg = "Lavender", fg = "black")
    e5.pack()
    e5.place(relx=0.5,rely=0.50, anchor=CENTER)
    fx=tkinter.Entry(windowsecante)
    fx.pack()
    fx.place(relx=0.5,rely=0.60, anchor=CENTER)
    botonCal=tkinter.Button(windowsecante,text="Calcular", bg="Gainsboro",command=lambda:secante(fx.get(),xi.get(),xf.get(),n.get(),t.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.80, anchor=CENTER) 

def openwindowRaices():
    windowRaiz=Toplevel(mainW)
    windowRaiz.title("Newton")
    windowRaiz.geometry("200x200")
    windowRaiz.config(bg="Lavender")
    e1 = tkinter.Label(windowRaiz, text = "x inicial o x0", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.50, anchor=CENTER)
    x0=tkinter.Entry(windowRaiz)
    x0.pack()
    x0.place(relx=0.2,rely=0.60, anchor=CENTER, width = 40)
    e1 = tkinter.Label(windowRaiz, text = "Tolerancia", bg = "Lavender", fg = "black")
    e1.pack()
    e1.place(relx=0.2,rely=0.20, anchor=CENTER)
    t=tkinter.Entry(windowRaiz)
    t.pack()
    t.place(relx=0.2,rely=0.30, anchor=CENTER, width = 40)
    e3 = tkinter.Label(windowRaiz, text = "Número iter", bg = "Lavender", fg = "black")
    e3.pack()
    e3.place(relx=0.8,rely=0.50, anchor=CENTER)
    n=tkinter.Entry(windowRaiz)
    n.pack()
    n.place(relx=0.8,rely=0.60, anchor=CENTER, width = 40)
    e4 = tkinter.Label(windowRaiz, text = "Función", bg = "Lavender", fg = "black")
    e4.pack()
    e4.place(relx=0.8,rely=0.20, anchor=CENTER)
    fx=tkinter.Entry(windowRaiz)
    fx.pack()
    fx.place(relx=0.8,rely=0.30, anchor=CENTER, width = 40)
    botonCal=tkinter.Button(windowRaiz,text="Calcular", bg="Gainsboro",command=lambda:newton(fx.get(),x0.get(),n.get(),t.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.90, anchor=CENTER) 

def openWindowEcu():
    windowEcu=Toplevel(mainW)
    windowEcu.title("Ecuaciones de una variable")
    windowEcu.geometry("300x300")
    windowEcu.config(bg="LavenderBlush")

    botonBus=tkinter.Button(windowEcu,text="Búsquedas incrementales", bg="Thistle", command = openwindowBusquedas)
    botonBus.pack()
    botonBus.place(relx=0.5,rely=0.125, anchor=CENTER)

    botonBis=tkinter.Button(windowEcu,text="Método de bisección", bg="Thistle", command = openwindowBisecciones)
    botonBis.pack()
    botonBis.place(relx=0.5,rely=0.25, anchor=CENTER)

    botonPunF=tkinter.Button(windowEcu,text="Método de punto fijo", bg="Thistle", command = openwindowPuntoF)
    botonPunF.pack()
    botonPunF.place(relx=0.5,rely=0.375, anchor=CENTER)

    botonRegF=tkinter.Button(windowEcu,text="Método de la regla falsa", bg="Thistle", command = openwindowReglaf)
    botonRegF.pack()
    botonRegF.place(relx=0.5,rely=0.50, anchor=CENTER)

    botonNew=tkinter.Button(windowEcu,text="Método de Newton", bg="Thistle", command = openwindowNewton)
    botonNew.pack()
    botonNew.place(relx=0.5,rely=0.625, anchor=CENTER)

    botonSec=tkinter.Button(windowEcu,text="Método de la secante", bg="Thistle", command = openwindowSecante)
    botonSec.pack()
    botonSec.place(relx=0.5,rely=0.75, anchor=CENTER)

    botonRaiM=tkinter.Button(windowEcu,text="Método de raíces múltiples", bg="Thistle" ,command = openwindowRaices)
    botonRaiM.pack()
    botonRaiM.place(relx=0.5,rely=0.875, anchor=CENTER)


def openwindowGauss():
    windowGauss=Toplevel(mainW)
    windowGauss.title("Eliminación Gausiana")
    windowGauss.geometry("350x200")
    windowGauss.config(bg="Lavender")
    e1 = tkinter.Label(windowGauss, text = "Número de ecuaciones:", bg = "Lavender", fg= "black")
    e1.pack()
    e1.place(relx=0.5,rely=0.15, anchor=CENTER)
    num=tkinter.Entry(windowGauss)
    num.pack()
    num.place(relx=0.5,rely=0.25, anchor=CENTER)
    e2 = tkinter.Label(windowGauss, text = "Matriz aumentada:", bg = "Lavender", fg= "black")
    e2.pack()
    e2.place(relx=0.5,rely=0.40, anchor=CENTER)
    mat=tkinter.Entry(windowGauss)
    mat.pack()
    mat.place(relx=0.5,rely=0.50, anchor=CENTER)
    e3 = tkinter.Label(windowGauss, text = "*Elementos separados por , y filas separadas por ;", bg = "Lavender", fg= "black")
    e3.pack()
    e3.place(relx=0.5,rely=0.60, anchor=CENTER)
    botonCal=tkinter.Button(windowGauss,text="Calcular", bg="Gainsboro",command=lambda:gaussJordan(num.get(),mat.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.75, anchor=CENTER)


def openwindowParcial():
    windowParcial=Toplevel(mainW)
    windowParcial.title("Pivoteo Parcial")
    windowParcial.geometry("350x200")
    windowParcial.config(bg="Lavender")
    e2 = tkinter.Label(windowParcial, text = "Matriz aumentada:", bg = "Lavender", fg= "black")
    e2.pack()
    e2.place(relx=0.5,rely=0.30, anchor=CENTER)
    mat=tkinter.Entry(windowParcial)
    mat.pack()
    mat.place(relx=0.5,rely=0.40, anchor=CENTER)
    e3 = tkinter.Label(windowParcial, text = "*Elementos separados por , y filas separadas por ;", bg = "Lavender", fg= "black")
    e3.pack()
    e3.place(relx=0.5,rely=0.50, anchor=CENTER)
    botonCal=tkinter.Button(windowParcial,text="Calcular", bg="Gainsboro",command=lambda:PivParcial(mat.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.65, anchor=CENTER)

def openwindowTotal():
    windowTotal=Toplevel(mainW)
    windowTotal.title("Pivoteo Total")
    windowTotal.geometry("350x200")
    windowTotal.config(bg="Lavender")
    e2 = tkinter.Label(windowTotal, text = "Matriz aumentada:", bg = "Lavender", fg= "black")
    e2.pack()
    e2.place(relx=0.5,rely=0.30, anchor=CENTER)
    mat=tkinter.Entry(windowTotal)
    mat.pack()
    mat.place(relx=0.5,rely=0.40, anchor=CENTER)
    e3 = tkinter.Label(windowTotal, text = "*Elementos separados por , y filas separadas por ;", bg = "Lavender", fg= "black")
    e3.pack()
    e3.place(relx=0.5,rely=0.50, anchor=CENTER)
    botonCal=tkinter.Button(windowTotal,text="Calcular", bg="Gainsboro",command=lambda:PivTotal(mat.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.65, anchor=CENTER)
   
def openwindowLU():
    windowLU=Toplevel(mainW)
    windowLU.title("Factorización LU")
    windowLU.geometry("350x200")
    windowLU.config(bg="Lavender")
    e2 = tkinter.Label(windowLU, text = "Matriz de coeficientes:", bg = "Lavender", fg= "black")
    e2.pack()
    e2.place(relx=0.5,rely=0.20, anchor=CENTER)
    mat=tkinter.Entry(windowLU)
    mat.pack()
    mat.place(relx=0.5,rely=0.30, anchor=CENTER)
    e3 = tkinter.Label(windowLU, text = "Matriz de términos independientes:", bg = "Lavender", fg= "black")
    e3.pack()
    e3.place(relx=0.5,rely=0.40, anchor=CENTER)
    ind=tkinter.Entry(windowLU)
    ind.pack()
    ind.place(relx=0.5,rely=0.50, anchor=CENTER)
    e4 = tkinter.Label(windowLU, text = "*Elementos separados por , y filas separadas por ;", bg = "Lavender", fg= "black")
    e4.pack()
    e4.place(relx=0.5,rely=0.60, anchor=CENTER)
    botonCal=tkinter.Button(windowLU,text="Calcular", bg="Gainsboro",command=lambda:LU(mat.get(), ind.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.75, anchor=CENTER)

def openwindowJacobi():
    windowJacob=Toplevel(mainW)
    windowJacob.title("Método Jacobi")
    windowJacob.geometry("350x200")
    windowJacob.config(bg="Lavender")
    e2 = tkinter.Label(windowJacob, text = "Matriz de coeficientes:", bg = "Lavender", fg= "black")
    e2.pack()
    e2.place(relx=0.5,rely=0.10, anchor=CENTER)
    mat=tkinter.Entry(windowJacob)
    mat.pack()
    mat.place(relx=0.5,rely=0.20, anchor=CENTER)
    e3 = tkinter.Label(windowJacob, text = "Matriz de términos independientes:", bg = "Lavender", fg= "black")
    e3.pack()
    e3.place(relx=0.5,rely=0.30, anchor=CENTER)
    ind=tkinter.Entry(windowJacob)
    ind.pack()
    ind.place(relx=0.5,rely=0.40, anchor=CENTER)
    e5 = tkinter.Label(windowJacob, text = "Número iter", bg = "Lavender", fg= "black")
    e5.pack()
    e5.place(relx=0.5,rely=0.50, anchor=CENTER)
    n=tkinter.Entry(windowJacob)
    n.pack()
    n.place(relx=0.5,rely=0.60, anchor=CENTER)
    e4 = tkinter.Label(windowJacob, text = "*Elementos separados por , y filas separadas por ;", bg = "Lavender", fg= "black")
    e4.pack()
    e4.place(relx=0.5,rely=0.70, anchor=CENTER)
    botonCal=tkinter.Button(windowJacob,text="Calcular", bg="Gainsboro",command=lambda:jacobi(mat.get(), ind.get(),n.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.85, anchor=CENTER)

def openwindowSeidel():
    windowJacob=Toplevel(mainW)
    windowJacob.title("Método Jacobi")
    windowJacob.geometry("350x200")
    windowJacob.config(bg="Lavender")
    e2 = tkinter.Label(windowJacob, text = "Matriz de coeficientes:", bg = "Lavender", fg= "black")
    e2.pack()
    e2.place(relx=0.5,rely=0.10, anchor=CENTER)
    mat=tkinter.Entry(windowJacob)
    mat.pack()
    mat.place(relx=0.5,rely=0.20, anchor=CENTER)
    e3 = tkinter.Label(windowJacob, text = "Matriz de términos independientes:", bg = "Lavender", fg= "black")
    e3.pack()
    e3.place(relx=0.5,rely=0.30, anchor=CENTER)
    ind=tkinter.Entry(windowJacob)
    ind.pack()
    ind.place(relx=0.5,rely=0.40, anchor=CENTER)
    e5 = tkinter.Label(windowJacob, text = "Número iter", bg = "Lavender", fg= "black")
    e5.pack()
    e5.place(relx=0.5,rely=0.50, anchor=CENTER)
    n=tkinter.Entry(windowJacob)
    n.pack()
    n.place(relx=0.5,rely=0.60, anchor=CENTER)
    e4 = tkinter.Label(windowJacob, text = "*Elementos separados por , y filas separadas por ;", bg = "Lavender", fg= "black")
    e4.pack()
    e4.place(relx=0.5,rely=0.70, anchor=CENTER)
    botonCal=tkinter.Button(windowJacob,text="Calcular", bg="Gainsboro",command=lambda:seidel(mat.get(), ind.get(),n.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.85, anchor=CENTER)

def openWindowSiste():
    windowSis=Toplevel(mainW)
    windowSis.title("Sistemas de ecuaciones")
    windowSis.geometry("300x300")
    windowSis.config(bg="LavenderBlush")

    botonGaus=tkinter.Button(windowSis,text="Eliminación Gaussiana", bg="Thistle",command=openwindowGauss)
    botonGaus.pack()
    botonGaus.place(relx=0.5,rely=0.143, anchor=CENTER)

    botonPivP=tkinter.Button(windowSis,text="Pivoteo Parcial", bg="Thistle", command = openwindowParcial)
    botonPivP.pack()
    botonPivP.place(relx=0.5,rely=0.286, anchor=CENTER)

    botonPivT=tkinter.Button(windowSis,text="Pivoteo Total", bg="Thistle", command = openwindowTotal)
    botonPivT.pack()
    botonPivT.place(relx=0.5,rely=0.429, anchor=CENTER)

    botonLU=tkinter.Button(windowSis,text="Factorización LU", bg="Thistle", command = openwindowLU)
    botonLU.pack()
    botonLU.place(relx=0.5,rely=0.572, anchor=CENTER)

    botonJa=tkinter.Button(windowSis,text="Método de Jacobi", bg="Thistle", command = openwindowJacobi)
    botonJa.pack()
    botonJa.place(relx=0.5,rely=0.715, anchor=CENTER)

    botonGausSei=tkinter.Button(windowSis,text="Gauss-Seidel", bg="Thistle", command = openwindowSeidel)
    botonGausSei.pack()
    botonGausSei.place(relx=0.5,rely=0.858, anchor=CENTER)


def openwindowLagrage():
    windowLa=Toplevel(mainW)
    windowLa.title("Lagrage")
    windowLa.geometry("200x300")
    windowLa.config(bg="Lavender")
    Lx=tkinter.Label( windowLa, text="Coordenadas en x", bg = "Lavender", fg ="black")
    Lx.pack()
    Lx.place(relx=0.5,rely=0.1, anchor=CENTER)
    x=tkinter.Entry(windowLa)
    x.pack()
    x.place(relx=0.5,rely=0.2, anchor=CENTER)
    Ly=tkinter.Label( windowLa, text="Coordenadas en y", bg = "Lavender", fg ="black")
    Ly.pack()
    Ly.place(relx=0.5,rely=0.3, anchor=CENTER)
    y=tkinter.Entry(windowLa)
    y.pack()
    y.place(relx=0.5,rely=0.4, anchor=CENTER)
    Lint=tkinter.Label( windowLa, text="Punto de interpolación", bg = "Lavender", fg ="black")
    Lint.pack()
    Lint.place(relx=0.5,rely=0.5, anchor=CENTER)
    int=tkinter.Entry(windowLa)
    int.pack()
    int.place(relx=0.5,rely=0.6, anchor=CENTER)
    botonCal=tkinter.Button(windowLa,text="Calcular",bg="Gainsboro",command=lambda:lagrange(x.get(),y.get(),int.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.9, anchor=CENTER)

def openwindowSplineQ():
    windowSplineQ=Toplevel(mainW)
    windowSplineQ.title("Lagrage")
    windowSplineQ.geometry("200x300")
    windowSplineQ.config(bg="Lavender")
    Lx=tkinter.Label( windowSplineQ, text="Coordenadas en x", bg = "Lavender", fg = 'black')
    Lx.pack()
    Lx.place(relx=0.5,rely=0.1, anchor=CENTER)
    x=tkinter.Entry(windowSplineQ)
    x.pack()
    x.place(relx=0.5,rely=0.2, anchor=CENTER)
    Ly=tkinter.Label( windowSplineQ, text="Coordenadas en y", bg = "Lavender", fg = 'black')
    Ly.pack()
    Ly.place(relx=0.5,rely=0.3, anchor=CENTER)
    y=tkinter.Entry(windowSplineQ)
    y.pack()
    y.place(relx=0.5,rely=0.4, anchor=CENTER)
    Lint=tkinter.Label( windowSplineQ, text="Punto de interpolación", bg = "Lavender", fg = 'black')
    Lint.pack()
    Lint.place(relx=0.5,rely=0.5, anchor=CENTER)
    int=tkinter.Entry(windowSplineQ)
    int.pack()
    int.place(relx=0.5,rely=0.6, anchor=CENTER)
    botonCal=tkinter.Button(windowSplineQ,text="Calcular",bg="Gainsboro",command=lambda:quadSplines(x.get(),y.get(),int.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.9, anchor=CENTER)


def openwindowDiff():
    windowDiff=Toplevel(mainW)
    windowDiff.title("Diferencias divididas")
    windowDiff.geometry("200x300")
    windowDiff.config(bg="Lavender")
    Lx=tkinter.Label( windowDiff, text="Coordenadas en x", bg = "Lavender", fg = "black")
    Lx.pack()
    Lx.place(relx=0.5,rely=0.1, anchor=CENTER)
    x=tkinter.Entry(windowDiff)
    x.pack()
    x.place(relx=0.5,rely=0.2, anchor=CENTER)
    Ly=tkinter.Label( windowDiff, text="Coordenadas en y", bg = "Lavender", fg = "black")
    Ly.pack()
    Ly.place(relx=0.5,rely=0.3, anchor=CENTER)
    y=tkinter.Entry(windowDiff)
    y.pack()
    y.place(relx=0.5,rely=0.4, anchor=CENTER)
    Lint=tkinter.Label( windowDiff, text="Punto de interpolación", bg = "Lavender", fg = "black")
    Lint.pack()
    Lint.place(relx=0.5,rely=0.5, anchor=CENTER)
    int=tkinter.Entry(windowDiff)
    int.pack()
    int.place(relx=0.5,rely=0.6, anchor=CENTER)
    botonCal=tkinter.Button(windowDiff,text="Calcular",bg="Gainsboro", command=lambda:newton_polynomial(x.get(),y.get(),int.get()))
    botonCal.pack()
    botonCal.place(relx=0.5,rely=0.70, anchor=CENTER)

def openWindowInt():
     windowInt=Toplevel(mainW)
     windowInt.title("Interpolación")
     windowInt.geometry("300x300")
     windowInt.config(bg="LavenderBlush")
    
     botonDifD=tkinter.Button(windowInt,text="Diferencias divididas", bg="Thistle", command =openwindowDiff)
     botonDifD.pack()
     botonDifD.place(relx=0.5,rely=0.25, anchor=CENTER)

     botonSisEcu=tkinter.Button(windowInt,text="Lagrange", bg="Thistle", command = openwindowLagrage)
     botonSisEcu.pack()
     botonSisEcu.place(relx=0.5,rely=0.5, anchor=CENTER)

     botonInt=tkinter.Button(windowInt,text="Splines cuadráticas", bg="Thistle", command = openwindowSplineQ)
     botonInt.pack()
     botonInt.place(relx=0.5,rely=0.75, anchor=CENTER)





botonEcu=tkinter.Button(mainW,text="Ecuaciones de una variable", bg="misty rose",command=openWindowEcu)
botonEcu.pack()
botonEcu.place(relx=0.5,rely=0.25, anchor=CENTER)

botonSisEcu=tkinter.Button(mainW,text="Sistemas de ecuaciones", bg="misty rose",command=openWindowSiste)
botonSisEcu.pack()
botonSisEcu.place(relx=0.5,rely=0.5, anchor=CENTER)

botonInt=tkinter.Button(mainW,text="Interpolación", bg="misty rose",command=openWindowInt)
botonInt.pack()
botonInt.place(relx=0.5,rely=0.75, anchor=CENTER)



mainW.mainloop()
