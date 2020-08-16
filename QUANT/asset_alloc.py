# 用于整理资产配置相关的理论和框架；同时用于组合自己的策略方法
# 请务必测试通过并确保稳健性之后再编辑为函数
import numpy as np
import pandas as pd


# 投资组合再平衡
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

# 基本模型：Markowitz、BL等等

def BL_Model():
    return 0
    
def Markowitz():
    return 0

