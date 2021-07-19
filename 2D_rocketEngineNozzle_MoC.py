#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:19:10 2021

@author: karlo
"""
import math
import numpy as np
import matplotlib.pyplot as plt


class rocketEngineNozzle_MoC():
    def __init__(self,M_e,n,D_t):  
        self.M_e = M_e                          #design exit Mach number
        self.n = n                              #number of characteristics
        self.D_t = D_t                          #Throat diameter at inlet 
        
        self.g = 1.4                              
        self.R = 287.1      
        self.k1 = 1+((self.g-1)/2*self.M_e**2)
        self.k2 = (self.g+1)/(2*(self.g-1))
        self.r = D_t/2
        self.nu_e = math.sqrt((self.g+1)/(self.g-1))*math.atan(math.sqrt((self.g-1) \
                /(self.g+1)*(M_e**2-1)))-math.atan(math.sqrt(M_e**2-1))
        self.nu_ed = self.nu_e*180/math.pi
        self.thetaMax = self.nu_e/2
        self.theta_start = np.linspace(0.001,self.thetaMax,n)
        self.thetaD = self.theta_start*180/np.pi
        self.Aratio = 1/M_e*(2*self.k1/(self.g+1))**self.k2
        self.D_e = self.Aratio*D_t
        self.N = int(n*(n+3)/2)
        
        self.nui = np.zeros((self.n,self.n))
        self.thetai = np.zeros((self.n,self.n))
        self.C_Ii = np.zeros((self.n,self.n))
        self.C_IIi = np.zeros((self.n,self.n))
        
        self.nub = np.zeros((1,self.n))
        self.thetab = np.zeros((1,self.n))
        self.C_Ib = np.zeros((1,self.n))
        self.C_IIb = np.zeros((1,self.n))
               
       
           
        self.thetai[:,0] = self.theta_start
        self.nui[:,0] = self.thetai[:,0]
        self.C_Ii[:,0] = 2*self.nui[:,0]
        self.C_IIi[:,0] = self.nui[:,0]-self.thetai[:,0]
        for i in range(self.n-1):
            for j in range(self.n-1-i):
                if j == 0:
                  self.thetai[j,i+1] = 0
                  self.nui[j,i+1] = self.C_Ii[j+1,i]
                  self.C_Ii[j,i+1] = self.nui[j,i+1]+self.thetai[j,i+1]
                  self.C_IIi[j,i+1] = self.nui[j,i+1]-self.thetai[j,i+1]
                else:
                  self.nui[j,i+1] = 1/2*(self.C_Ii[j+1,i]+self.C_IIi[j-1,i+1])  
                  self.thetai[j,i+1] = 1/2*(self.C_Ii[j+1,i]-self.C_IIi[j-1,i+1]) 
                  self.C_Ii[j,i+1] = self.nui[j,i+1]+self.thetai[j,i+1]
                  self.C_IIi[j,i+1] = self.nui[j,i+1]-self.thetai[j,i+1]              
        for i in range(self.n):
            self.thetab[0,i] = self.thetai[self.n-1-i,i]
            self.C_IIb[0,i] = self.C_IIi[self.n-1-i,i]
            self.nub[0,i] = self.thetab[0,i]+self.C_IIb[0,i]
            self.C_Ib[0,i] = self.thetab[0,i]+self.nui[0,i]
            
        z1 = np.zeros((1,self.n))
        self.nu= np.vstack((self.nui,z1))
        self.theta= np.vstack((self.thetai,z1))
        self.C_I= np.vstack((self.C_Ii,z1))
        self.C_II= np.vstack((self.C_IIi,z1))
        
        z2 = np.zeros((self.n+1,1))
        self.nu= np.hstack((self.nu,z2))
        self.theta= np.hstack((self.theta,z2))
        self.C_I= np.hstack((self.C_I,z2))
        self.C_II= np.hstack((self.C_II,z2))
        
        for i in range(self.n):
            self.nu[self.n-i,i] = self.nub[0,i]
            self.theta[self.n-i,i] = self.thetab[0,i] 
            self.C_I[self.n-i,i] = self.C_Ib[0,i] 
            self.C_II[self.n-i,i] = self.C_IIb[0,i]
        self.thetad = self.theta*180/math.pi
 #######################        
    
        self.M = np.zeros((self.n+1,self.n+1))
        self.mi = np.zeros((self.n+1,self.n+1))
        self.theta_p_mi = np.zeros((self.n+1,self.n+1))
        self.theta_m_mi = np.zeros((self.n+1,self.n+1))
        self.fun = np.zeros((self.n+1,self.n+1))
        self.fun1 = np.zeros((self.n+1,self.n+1))
        self.fun2 = np.zeros((self.n+1,self.n+1))
        self.dfun_dM = np.zeros((self.n+1,self.n+1))
        
        for i in range(self.n+1):
            for j in range(self.n+1):
                if (j>self.n-i):
                    continue
                elif (i == self.n):
                    continue
                self.M_0 = 1.05
                self.g1 = (self.g+1)/(self.g-1)
                self.dM = 0.01
                self.fun[j,i] = (math.sqrt(self.g1)*math.atan(math.sqrt(self.g1**(-1)*\
                              (self.M_0**2-1)))-math.atan(math.sqrt(self.M_0**2-1)))\
                               -self.nu[j,i]
                
                while (abs(self.fun[j,i]) > 1e-5):
                    self.M1 = self.M_0+self.dM
                    self.M2 = self.M_0-self.dM
                    self.fun1[j,i] = (math.sqrt(self.g1)*math.atan(math.sqrt(self.g1**(-1)*\
                              (self.M1**2-1)))-math.atan(math.sqrt(self.M1**2-1)))\
                               -self.nu[j,i]
                    self.fun2[j,i] = (math.sqrt(self.g1)*math.atan(math.sqrt(self.g1**(-1)*\
                              (self.M2**2-1)))-math.atan(math.sqrt(self.M2**2-1)))\
                               -self.nu[j,i]       
                    self.dfun_dM[j,i] = (self.fun1[j,i]-self.fun2[j,i])/(2*self.dM)
                    self.fun[j,i] = (math.sqrt(self.g1)*math.atan(math.sqrt(self.g1**(-1)*\
                              (self.M_0**2-1)))-math.atan(math.sqrt(self.M_0**2-1)))\
                               -self.nu[j,i]
                    self.M[j,i] = self.M_0-(self.fun[j,i]/self.dfun_dM[j,i])
                    self.M_0 = self.M[j,i]
                    self.mi[j,i] = math.asin(1/self.M[j,i])
                    self.theta_p_mi[j,i] = self.theta[j,i]+self.mi[j,i]
                    self.theta_m_mi[j,i] = self.theta[j,i]-self.mi[j,i]
        self.theta_p_mid = self.theta_p_mi*180/math.pi
        self.theta_m_mid = self.theta_m_mi*180/math.pi
        self.mid = self.mi*180/math.pi
####################      

        self.xp = np.zeros((self.n+1,self.n+1))
        self.yp = np.zeros((self.n+1,self.n+1))
        self.m_I = np.zeros((self.n+1,self.n+1))
        self.m_II = np.zeros((self.n+1,self.n+1))
        
        for i in range(self.n+1):
            for j in range(self.n+1):
                if (j>self.n-i):
                    continue
                elif (i == self.n):
                    continue  
                elif (i == 0 and j<self.n):
                    self.m_I[j,i] = math.tan(self.theta_m_mi[j,i])
                elif (i == 0 and j == self.n):
                    self.m_I[j,i] = math.tan(self.theta[j,i])
                elif (j == self.n-i and i>0):
                    self.m_I[j,i] = math.tan(1/2*(self.theta[j,i]\
                                                  +self.theta[j+1,i-1]))
                else:
                    self.m_I[j,i] = math.tan(1/2*(self.theta_m_mi[j,i]\
                                                  +self.theta_m_mi[j+1,i-1]))
        for i in range(self.n+1):
            for j in range(self.n+1):
                if (j>self.n-i):
                    continue
                elif (i == self.n):
                    continue  
                if (j == 0):
                    self.m_II[j,i] = math.tan(self.theta_p_mi[j,i])
                else:
                    self.m_II[j,i] = math.tan(1/2*(self.theta_p_mi[j,i]\
                                                   +self.theta_p_mi[j-1,i]))
        for i in range(self.n+1):
            for j in range(self.n+1):
                self.x0 = 0
                self.y0 = self.r
                if (j>self.n-i):
                    continue
                elif (i == self.n):
                    continue 
                if (j == 0 and i == 0):
                    self.xp[j,i] = self.x0-self.y0/self.m_I[j,i]
                    self.yp[j,i] = 0
                elif (j > 0 and i == 0):
                    self.xp[j,i] = (self.y0-self.yp[j-1,i]+self.m_II[j,i]*self.xp[j-1,i]\
                                    -self.m_I[j,i]*self.x0)/(self.m_II[j,i]-self.m_I[j,i])
                    self.yp[j,i] = self.y0+self.m_I[j,i]*(self.xp[j,i]-self.x0)
                elif (j == 0 and i>0):
                    self.yp[j,i] = 0
                    self.xp[j,i] = (self.yp[j+1,i-1]-((-1)*self.yp[j+1,i-1])\
                                    +((-1)*self.m_I[j,i])*self.xp[j+1,i-1]\
                                    -self.m_I[j,i]*self.xp[j+1,i-1])/((-1)\
                                    *self.m_I[j,i]-self.m_I[j,i])
                elif(j>0 and i >0):
                    self.xp[j,i] = (self.yp[j+1,i-1]-self.yp[j-1,i]+self.m_II[j,i]*self.xp[j-1,i]\
                                    -self.m_I[j,i]*self.xp[j+1,i-1])/(self.m_II[j,i]-self.m_I[j,i])
                    self.yp[j,i] = self.yp[j+1,i-1]+self.m_I[j,i]*(self.xp[j,i]-self.xp[j+1,i-1])
        self.Aratio1 = self.yp[1,self.n-1]/self.y0
    def plotting(self):  
        plt.figure()
        plt.grid(True)
        plt.plot([self.x0,self.xp[self.n,0]],[self.y0,self.yp[self.n,0]],'-bo')
        plt.plot([self.xp[1,0],self.xp[0,1]],[self.yp[1,0],self.yp[0,1]],'--bo')
        for i in range(self.n+1):
            if (i<self.n):
                plt.plot([self.x0,self.xp[i,0]],[self.y0,self.yp[i,0]],'--bo')
            for j in range(self.n+1):
                if (j>self.n-i):
                    continue
                elif (i == self.n):
                    continue
                elif (j<self.n-i):
                    plt.plot([self.xp[j,i],self.xp[j+1,i]],[self.yp[j,i],self.yp[j+1,i]],'--bo')
                elif (j == self.n-i and i<self.n-1):
                    plt.plot([self.xp[j,i],self.xp[j-1,i+1]],[self.yp[j,i],self.yp[j-1,i+1]],'-bo')
        for i in range(self.n+1):
            for j in range(i):
                if (i>1 and i<self.n):
                    plt.plot([self.xp[i-j,j],self.xp[i-1-j,j+1]],\
                             [self.yp[i-j,j],self.yp[i-1-j,j+1]],'--bo')
                    
       
# =============================================================================
#                  
#         self.ypf = -1*self.yp
#         self.y0f = -1*self.y0
#         plt.plot([self.x0,self.xp[self.n,0]],[self.y0f,self.ypf[self.n,0]],'-bo')
#         plt.plot([self.xp[1,0],self.xp[0,1]],[self.ypf[1,0],self.ypf[0,1]],'--bo')
#         for i in range(self.n+1):
#             if (i<self.n):
#                 plt.plot([self.x0,self.xp[i,0]],[self.y0f,self.ypf[i,0]],'--bo')
#             for j in range(self.n+1):
#                 if (j>self.n-i):
#                     continue
#                 elif (i == self.n):
#                     continue
#                 elif (j<self.n-i):
#                     plt.plot([self.xp[j,i],self.xp[j+1,i]],[self.ypf[j,i],self.ypf[j+1,i]],'--bo')
#                 elif (j == self.n-i and i<self.n-1):
#                     plt.plot([self.xp[j,i],self.xp[j-1,i+1]],[self.ypf[j,i],self.ypf[j-1,i+1]],'-bo')
#         for i in range(self.n+1):
#             for j in range(i):
#                 if (i>1 and i<self.n):
#                     plt.plot([self.xp[i-j,j],self.xp[i-1-j,j+1]],\
#                              [self.ypf[i-j,j],self.ypf[i-1-j,j+1]],'--bo')
# =============================================================================
        ###########                
        self.Mc = np.zeros((self.n,1))
        self.xc = np.zeros((self.n,1))
        for i in range(self.n):
            self.Mc[i] = self.M[0,i]
            self.xc[i] = self.xp[0,i]
        self.Mc = np.append(self.Mc,self.M[1,self.n-1])
        self.xc = np.append(self.xc,self.xp[1,self.n-1])
        
        plt.figure()
        plt.grid(True)
        plt.plot(self.xc,self.Mc)
        
        
        
        ###########
        
            
            
        
        
                

c = rocketEngineNozzle_MoC(3, 20, 0.2)
c.plotting()
