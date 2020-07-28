# a library used to quantify frameworks and methods
# never try to build all the wheels yourself
import numpy as np
import pandas as pd


# Rebalance Function 
def Portfolio_Rebalance(Current_Position,Target_Perct=[],Hist_Return,Rebalance_Method='Reverse',Rebalance_Horizon='1y',Evaluate_Horizon='8y',Change_Perct=0):
    if (Rebalance_Method=='Reverse'):
        if Standard_Perct==[]:
            print('当Rebalance Method为Reverse时，必须指定目标比例Target Percentage!')
        Rebalance_Perct=Target_Perct

    
    elif (Rebalance_Method=='Momentum'):
        if (Change_Perct==0):
            print('当Rebalance Method为Momentum时，必须指定调整仓位比例Change_Perct!')

    else:
        print('需指定再平衡策略，或指定的再平衡策略不存在！')
    
    return (Rebalance_Perct, Rebalance_Position,Change_Position)

# Provide optimized portfolio according to models
def Optimized_Portfolio_Perct(Optim_Method='Markowitz',Target_Anual_Return=0.1,Target_Volatility=0.03,Correlation_Matrix):
    return 0

def BL_Model():
    return 0
    
def Markowitz():
    return 0

