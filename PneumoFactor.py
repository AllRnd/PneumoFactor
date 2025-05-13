import tkinter as tk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os 


def initializing_page():
    for widget in Dott.winfo_children():
        widget.destroy()
    root.geometry("650x300")
    Dott.pack(fill=tk.X)
    for name in factors:
        tk.Label(Dott, text=name, font=label_font).grid(row=0, column=factors.index(name), padx=5, pady=5)
    for i in range(0,len(factors)):
        tk.Label(Dott, text='модель', font=label_font).grid(row=1, column=i, padx=5, pady=5)
    for i in range(0,len(factors)):
        for metric in _metrics:
            tk.Label(Dott, text=metric+" = "+str(_metrics_value[i][_metrics.index(metric)]), font=label_font).grid(row=2+_metrics.index(metric), column=i, padx=5, pady=5)
    tk.Button(Dott, text='Открыть', font=label_font, command=_4_factor).grid(row=6, column=0, padx=5, pady=5) 
    tk.Button(Dott, text='Открыть', font=label_font, command=_5_factor).grid(row=6, column=1, padx=5, pady=5)
    tk.Button(Dott, text='Открыть', font=label_font, command=_18_factor).grid(row=6, column=2, padx=5, pady=5)
def _18_factor():
    for widget in Dott.winfo_children():
        widget.destroy()
    root.geometry("1000x600")
    Dott.pack(fill=tk.X)
    entries.clear()
    for factor in _18_factors:
        tk.Label(Dott, text=factor, font=label_font).grid(row=(_18_factors.index(factor)//4)*2, column=_18_factors.index(factor)%4, padx=10, pady=10)
        entry18 = tk.Entry(Dott, font=entry_font)
        entry18.grid(row=1+(_18_factors.index(factor)//4)*2, column=_18_factors.index(factor)%4, padx=10, pady=10)
        entries.append(entry18)
    tk.Button(Dott, text='Назад', font=label_font, command=initializing_page).grid(row=0, column=4, padx=5, pady=5)
    tk.Button(Dott, text='Рассчитать', font=label_font, command=_predict_18_f).grid(row=10, column=1, padx=5, pady=5)
    tk.Label(Dott, text='Logit: Вероятность гибели:', font=label_font).grid(row=11, column=0, padx=5, pady=5)
    _logit_pred_label=tk.Label(Dott, text='', font=label_font)
    _logit_pred_label.grid(row=11, column=1, padx=5, pady=5)
    tk.Label(Dott, text='MLP: Вероятность гибели:', font=label_font).grid(row=12, column=0, padx=5, pady=5)
    _MLP_pred_label=tk.Label(Dott, text='', font=label_font)
    _MLP_pred_label.grid(row=12, column=1, padx=5, pady=5)
def _predict_18_f():
    input=[]
    b=[0.00838803, 0.39799107, 0.27629413, 0.31176781, -0.28951416, -0.14343993, 0.02313269, 0.38079808, -0.8201489, 0.1794579,
       0.18294034, -0.12988248, -0.00245626, -0.0877649, 0.09992229, 0.3421549, -0.11090888, 0.19907207, 0.50867411]
    for i in range(0,len(entries)):
        input.append(float(entries[i].get()))
    ex=b[0]
    for i in range(0,len(entries)):
        ex=ex+float(b[i+1])*float(input[i])
    logit=1/(1+np.exp(-ex))
    if logit<0.5:
        _logit_pred_label=tk.Label(Dott, text=str(round(100*logit,2))+' %', fg="green", font=label_font)
        _logit_pred_label.grid(row=11, column=1, padx=5, pady=5)
    else:
        _logit_pred_label=tk.Label(Dott, text=str(round(100*logit,2))+' %', fg="red", font=label_font)
        _logit_pred_label.grid(row=11, column=1, padx=5, pady=5)
    MLP_18 = Sequential()
    MLP_18.add(Dense(38, input_dim=18, activation='relu')) # input layer requires input_dim param
    MLP_18.add(Dense(37, activation='relu'))
    MLP_18.add(Dense(57, activation='relu'))
    MLP_18.add(Dense(76, activation='relu'))
    MLP_18.add(Dense(25, activation='relu')) 
    MLP_18.add(Dense(10, activation='relu')) #activation='sigmoid'

    MLP_18.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1
    MLP_18.load_weights(os.getcwd()+'\\checkpoint_MLP_18.model.keras')
    input_MLP=pd.DataFrame(input).transpose()
    MLP_pred=round(100*MLP_18.predict(input_MLP)[0][0], 2)
    if MLP_pred<50:
        _MLP_pred_label=tk.Label(Dott, text=str(MLP_pred)+' %', fg="green", font=label_font)
        _MLP_pred_label.grid(row=12, column=1, padx=5, pady=5)
    else:
        _MLP_pred_label=tk.Label(Dott, text=str(MLP_pred)+' %', fg="red", font=label_font)
        _MLP_pred_label.grid(row=12, column=1, padx=5, pady=5)
def _5_factor():
    for widget in Dott.winfo_children():
        widget.destroy()
    root.geometry("1000x320")
    Dott.pack(fill=tk.X)
    entries.clear()
    for factor in _5_factors:
        tk.Label(Dott, text=factor, font=label_font).grid(row=(_5_factors.index(factor)//4)*2, column=_5_factors.index(factor)%4, padx=10, pady=10)
        entry5 = tk.Entry(Dott, font=entry_font)
        entry5.grid(row=1+(_5_factors.index(factor)//4)*2, column=_5_factors.index(factor)%4, padx=10, pady=10)
        entries.append(entry5)
    tk.Button(Dott, text='Назад', font=label_font, command=initializing_page).grid(row=0, column=4, padx=5, pady=5)
    tk.Button(Dott, text='Рассчитать', font=label_font, command=_predict_5_f).grid(row=4, column=1, padx=5, pady=5)
    tk.Label(Dott, text='Logit: Вероятность гибели:', font=label_font).grid(row=5, column=0, padx=5, pady=5)
    tk.Label(Dott, text='MLP: Вероятность гибели:', font=label_font).grid(row=6, column=0, padx=5, pady=5)
def _predict_5_f():
    input=[]
    b=[53.70360296, -0.47884319, -0.06199687, -0.12260113, 0.61963881, 0.14443033]
    for i in range(0,len(entries)):
        input.append(float(entries[i].get()))
    ex=b[0]
    for i in range(0,len(entries)):
        ex=ex+b[i+1]*input[i]
    logit=1/(1+np.exp(-ex))
    if logit<0.5:
        _logit_pred_label=tk.Label(Dott, text=str(round(100*logit,2))+' %', fg="green", font=label_font)
        _logit_pred_label.grid(row=5, column=1, padx=5, pady=5)
    else:
        _logit_pred_label=tk.Label(Dott, text=str(round(100*logit,2))+' %', fg="red", font=label_font)
        _logit_pred_label.grid(row=5, column=1, padx=5, pady=5)
    MLP_5 = Sequential()
    MLP_5.add(Dense(12, input_dim=5, activation='relu')) # input layer requires input_dim param
    MLP_5.add(Dense(11, activation='relu'))
    MLP_5.add(Dense(18, activation='relu'))
    MLP_5.add(Dense(24, activation='relu'))
    MLP_5.add(Dense(8, activation='relu')) #activation='sigmoid'

    MLP_5.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1
    MLP_5.load_weights(os.getcwd()+'\\checkpoint_MLP_5.model.keras')
    input_MLP=pd.DataFrame(input).transpose()
    MLP_pred=round(100*MLP_5.predict(input_MLP)[0][0], 2)
    if MLP_pred<50:
        _MLP_pred_label=tk.Label(Dott, text=str(MLP_pred)+' %', fg="green", font=label_font)
        _MLP_pred_label.grid(row=6, column=1, padx=5, pady=5)
    else:
        _MLP_pred_label=tk.Label(Dott, text=str(MLP_pred)+' %', fg="red", font=label_font)
        _MLP_pred_label.grid(row=6, column=1, padx=5, pady=5)
def _4_factor():
    for widget in Dott.winfo_children():
        widget.destroy()
    root.geometry("1000x300")
    Dott.pack(fill=tk.X)
    entries.clear()
    for factor in _4_factors:
        tk.Label(Dott, text=factor, font=label_font).grid(row=(_4_factors.index(factor)//4)*2, column=_4_factors.index(factor)%4, padx=10, pady=10)
        entry4 = tk.Entry(Dott, font=entry_font)
        entry4.grid(row=1+(_4_factors.index(factor)//4)*2, column=_4_factors.index(factor)%4, padx=10, pady=10)
        entries.append(entry4)
    tk.Button(Dott, text='Назад', font=label_font, command=initializing_page).grid(row=0, column=4, padx=5, pady=5)
    tk.Button(Dott, text='Рассчитать', font=label_font, command=_predict_4_f).grid(row=2, column=1, padx=5, pady=5)
    tk.Label(Dott, text='Logit: Вероятность гибели:', font=label_font).grid(row=3, column=0, padx=5, pady=5)
    tk.Label(Dott, text='MLP: Вероятность гибели:', font=label_font).grid(row=4, column=0, padx=5, pady=5)
def _predict_4_f():
    input=[]
    b=[55.88155571, -0.50561613, -0.06374456, -0.10899855, 0.72048198]
    for i in range(0,len(entries)):
        input.append(float(entries[i].get()))
    ex=b[0]
    for i in range(0,len(entries)):
        ex=ex+b[i+1]*input[i]
    logit=1/(1+np.exp(-ex))
    if logit<0.5:
        _logit_pred_label=tk.Label(Dott, text=str(round(100*logit,2))+' %', fg="green", font=label_font)
        _logit_pred_label.grid(row=3, column=1, padx=5, pady=5)
    else:
        _logit_pred_label=tk.Label(Dott, text=str(round(100*logit,2))+' %', fg="red", font=label_font)
        _logit_pred_label.grid(row=3, column=1, padx=5, pady=5)
    MLP_4 = Sequential()
    MLP_4.add(Dense(10, input_dim=4, activation='relu')) # input layer requires input_dim param
    MLP_4.add(Dense(9, activation='relu'))
    MLP_4.add(Dense(15, activation='relu'))
    MLP_4.add(Dense(20, activation='relu'))
    MLP_4.add(Dense(7, activation='relu')) #activation='sigmoid'

    MLP_4.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1
    MLP_4.load_weights(os.getcwd()+'\\checkpoint_MLP_4.model.keras')
    input_MLP=pd.DataFrame(input).transpose()
    MLP_pred=round(100*MLP_4.predict(input_MLP)[0][0], 2)
    if MLP_pred<50:
        _MLP_pred_label=tk.Label(Dott, text=str(MLP_pred)+' %', fg="green", font=label_font)
        _MLP_pred_label.grid(row=4, column=1, padx=5, pady=5)
    else:
        _MLP_pred_label=tk.Label(Dott, text=str(MLP_pred)+' %', fg="red", font=label_font)
        _MLP_pred_label.grid(row=4, column=1, padx=5, pady=5)

_18_factors=['Дней болезни', 'Т тела', 'ЧДД', 'ЧСС', 'Систолическое АД', 'Диастолическое АД', 
             'Рост, см', 'Сатурация-воздух', 'Степень ДН', 'Лейкоциты, 10^9/л', 'Гемоглобин', 'MCV',
             'СОЭ', 'Натрий', 'Хлор', 'Белок', 'Фибриноген', 'Индекс Чарлсона']
_5_factors=['Сатурация-воздух', 'Гемоглобин', 'Белок', 'Фибриноген', 'Индекс Чарлсона']
_4_factors=['Сатурация-воздух', 'Гемоглобин', 'Белок', 'Фибриноген']
factors=['4-хфакторная модель', '5-ифакторная модель', '18-ифакторная модель']
_metrics=['Accuracy', 'Precision', 'Recall', 'F1']
_metrics_value=[[0.916, 0.863, 0.949, 0.904], [0.926, 0.913, 0.933, 0.923], [0.947, 0.943, 0.943, 0.943]]


root = tk.Tk()
root.title("Предсказание процента жира")
entries = []
label_font = ("Arial", 14) # Увеличение шрифта для надписей
entry_font = ("Arial", 12) # Увеличение шрифта для полей ввода

# Пол
Dott = tk.Frame()
initializing_page()

root.mainloop() 