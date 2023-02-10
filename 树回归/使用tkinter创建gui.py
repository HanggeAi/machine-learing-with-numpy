# 程序清单9-6 用于构建树管理器界面的tkinter小插件
from ast import Delete
import numpy as np
import tkinter as tk
from regTree import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a=reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN<2:
            tolN=2
        myTree=createTree(reDraw.rawDat,modelLeaf,modelErr,(tolS,tolN))
        yHat=createForeCast(myTree,reDraw.testDat,modelTreeEval)
    
    else:
        myTree=createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat=createForeCast(myTree,reDraw.testDat)
    
    reDraw.a.scatter(reDraw.rawDat[:,0],reDraw.rawDat[:,1],s=5)
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)
    reDraw.canvas.show()

def drawNewTree():
    return


root=tk.Tk()
tk.Label(root,text='Plot place holder').grid(row=0,columnspan=3)
tk.Label(root,text='tolN').grid(row=1,column=0)
tolNentry=tk.Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')

tk.Label(root,text='tolS').grid(row=2,column=0)
tolSentry=tk.Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')
tk.Button(root,text='ReDraw',command=drawNewTree).grid(row=1,column=2,rowspan=3)

chkBtnVar=tk.IntVar()
chkBtnm=tk.Checkbutton(root,text='Model Tree',variable=chkBtnVar)
chkBtnm.grid(row=3,column=0,columnspan=2)

reDraw.rawDat=np.mat(loadDataSet('D:\\机器学习实战代码\\machinelearninginaction\\Ch09\\sine.txt'))
reDraw.testDat=np.arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1,10)
root.mainloop()

reDraw.f=Figure(figsize=(5,4),dpi=100)
reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)


def getInputs():
    try:
        tolN=int(tolNentry.get())
    except:
        tolN=10
        print('enter integer for tolN')
        tolNentry.delete(0,tk.END)
        tolNentry.insert(0,'10')
    try:
        tolS=float(tolNentry.get())
    except:
        tolS=1.0
        print('enter integer for tolS')
        tolNentry.delete(0,tk.END)
        tolNentry.insert(0,'1.0')
    return tolN,tolS


def drawNewTree():
    tolN,tolS=getInputs()
    reDraw(tolS,tolN)