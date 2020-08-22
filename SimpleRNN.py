'''

Author: Bilgehan Kösem
E-mail: bilkos92@gmail.com
Date created: 07.18.2020
Date last modified: 07.18.2020
Python Version: 3.7
    
'''

import numpy as np
import copy

class SimpleRNN:
    def __init__(self,
                 layer_list,
                 time_steps,
                 weight_scaler = 1,
                 firing_rate_scaler = 0,
                 learning_rate = 0.1,
                 loss_function = 'mse',
                 optimizer = 'sgd'):
        
        #Assign basic parameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.shape = layer_list
        self.time_steps = time_steps
        self.loss_func = loss_function
        n = len(layer_list)
        self.dh_list = []
        self.gradient_list=[]
        
        #Initialize layers
        self.layers = []
        for i in range(n):
            self.layers.append(np.zeros(self.shape[i]))
        
        #Q and D are initiliazed with zeros
        self.Q_intime = []
        for t in range(self.time_steps+1):
            Q = []
            for i in range(n):
                Q.append(np.zeros((self.shape[i],1)))
            self.Q_intime.append(copy.deepcopy(Q))

        # Initialize vertical weights: W_vertical
        self.W_ver = []
        for i in range(n-1):
            self.W_ver.append(np.zeros((self.shape[i],self.shape[i+1])))
            self.gradient_list.append(np.zeros((self.shape[i],self.shape[i+1])))
        
        
        #Initialize horizontal wieghts: W_hor
        self.W_hor = np.zeros((self.shape[1],self.shape[1]))
        self.gradient_list.append(np.zeros((self.shape[1],self.shape[1])))

        self.bh = np.zeros((self.shape[1], 1))
        self.gradient_list.append(np.zeros((self.shape[1], 1)))
        
        self.by = np.zeros((self.shape[2], 1))
        self.gradient_list.append(np.zeros((self.shape[2], 1)))
        
        self.init_weights(weight_scaler)
        self.bp_counter = 0
        self.gradient_list_2=copy.deepcopy(self.gradient_list)
        self.gradient_list_3=copy.deepcopy(self.gradient_list)
        
    def init_weights(self,weight_scaler):
        
        for i in range(len(self.W_ver)):
            self.W_ver[i] = weight_scaler*np.random.rand(self.W_ver[i].shape[0],self.W_ver[i].shape[1])
        self.W_hor = weight_scaler*np.random.rand(self.W_hor.shape[0],self.W_hor.shape[1])
        
        return 1 
    
    def feedforward(self,input_list):

        t=0
        
        for t,input_t in enumerate(input_list,start = 1):
           
            self.Q_intime[t][0]=np.array(input_t).reshape(-1,1)
            self.Q_intime[t][1]=np.tanh(self.W_ver[0].transpose() @ self.Q_intime[t][0] + 
                    self.W_hor.transpose() @ self.Q_intime[t-1][1] +
                    self.bh)
        

        self.Q_intime[t][2] = self.W_ver[1].transpose()@self.Q_intime[t][1] + self.by
          
        return copy.deepcopy(self.Q_intime[t][2])
    
    def dh_dfi(self,t):
        return 1-self.Q_intime[t][1]**2
    
    def backpropagation(self,real_output,tmp1):
        self.bp_counter=self.bp_counter+1
        
        if self.optimizer=='nag':
            weights=np.asarray([self.W_ver[0],self.W_ver[1],self.W_hor,self.bh,self.by])

            x_ahead = weights-np.asarray(self.gradient_list)*0.9
            
            self.W_ver[0] = copy.deepcopy(x_ahead[0])
            self.W_ver[1] = copy.deepcopy(x_ahead[1])
            self.W_hor = copy.deepcopy(x_ahead[2])
            self.bh = copy.deepcopy(x_ahead[3])
            self.by = copy.deepcopy(x_ahead[4])
        
        loss = 0

        
        d_Wih = []
        d_Whh = []
        d_Who= []
        d_bh= []
        d_by = []
        
        #tmp1 = 1
        
        d_Who.append(tmp1*self.Q_intime[self.time_steps][1])
        d_by.append(tmp1)
        
        d_hidden_layer = copy.deepcopy(tmp1*self.W_ver[1])
        der_chain = d_hidden_layer * self.dh_dfi(self.time_steps)
        
        for t in reversed(range(1,self.time_steps+1)):#2 1 0
            
            d_Wih.append((der_chain @ self.Q_intime[t][0].transpose()).transpose())
            d_Whh.append((der_chain @ self.Q_intime[t][1].transpose()).transpose())
            d_bh.append(der_chain)
            self.dh_list.append(der_chain)
            der_chain = self.W_hor @ (self.dh_dfi(t-1) * der_chain)
            
        #dh_list.clear()
        #Create final gradients
        gradient_Wih = sum(d_Wih)
        gradient_Who = sum(d_Who)
        gradient_Whh = sum(d_Whh)
        gradient_bh = sum(d_bh)
        gradient_by = sum(d_by)
        
        for d in [gradient_Wih, gradient_Whh, gradient_Who, gradient_bh, gradient_by]:
          np.clip(d, -1, 1, out=d)
        
        #grads=np.asarray([sum(d_Wih),sum(d_Who),sum(d_Whh),sum(d_bh),sum(d_by)])
        grads=np.asarray([gradient_Wih,gradient_Who,gradient_Whh,gradient_bh,gradient_by])
        if self.optimizer == 'sgd':
            self.update_weigths(grads*self.learning_rate)

        elif self.optimizer == 'momentum':
            moment = np.asarray(self.gradient_list)*0.9
            grads = moment + grads*self.learning_rate
            self.update_weigths(grads)
            self.gradient_list=copy.deepcopy(grads)
        
        elif self.optimizer == 'nag':
            self.gradient_list = np.asarray(self.gradient_list)*0.9 + grads*self.learning_rate
            self.update_weigths(self.gradient_list)
            
        elif self.optimizer == 'adagrad':
            grads_2=np.square(grads)
            self.gradient_list+=copy.deepcopy(grads_2)
            for i in range(len(grads)):
                grads[i] = (self.learning_rate/(np.sqrt(self.gradient_list[i]+0.000001)))*grads[i]
            self.update_weigths(grads)
            #self.gradient_list+=copy.deepcopy(grads_2)
            
        elif self.optimizer == 'adadelta': #gradient_list_2->gt//gradient_list->teta_t

            eps=0.000001;beta=0.90;
            grads_2 = np.square(grads)
            self.gradient_list_2 = beta*np.asarray(self.gradient_list_2) + (1-beta)*grads_2
            
            delta_teta = copy.deepcopy(self.gradient_list)
            for i in range(len(grads)):
                #delta_teta[i] = (self.learning_rate/(np.sqrt(self.gradient_list_2[i]+0.000001)))*grads[i]
                delta_teta[i] = (np.sqrt(self.gradient_list[i]+0.000001)/(np.sqrt(self.gradient_list_2[i]+0.000001)))*grads[i]
      
            self.gradient_list = beta*np.asarray(self.gradient_list) + (1-beta)*np.square(delta_teta)

            for i in range(len(grads)):
                grads[i] = (np.sqrt(self.gradient_list[i]+0.000001)/(np.sqrt(self.gradient_list_2[i]+0.000001)))*grads[i]
            
            self.update_weigths(grads)
            #self.gradient_list = beta*np.asarray(self.gradient_list) + (1-beta)*np.square(grads)

            
        elif self.optimizer == 'rmsprop':
            eps=0.00000001
            grads_2 = np.square(grads)
            self.gradient_list = 0.9*np.asarray(self.gradient_list) + 0.1*grads_2

            for i in range(len(grads)):
                grads[i] = (self.learning_rate / np.sqrt(self.gradient_list[i]+eps)) * grads[i]
            self.update_weigths(grads)
        else:
            raise Exception('Unknown optimizer : \'{}\''.format(self.optimizer))
        
        return loss
    def update_weigths(self,grads):
        self.W_ver[0] = copy.deepcopy(self.W_ver[0] - grads[0])
        self.W_ver[1] = copy.deepcopy(self.W_ver[1] - grads[1])
        self.W_hor = copy.deepcopy(self.W_hor - grads[2])
        self.bh = copy.deepcopy(self.bh - grads[3])
        self.by = copy.deepcopy(self.by - grads[4])
    def softmax(self,xs):
      return np.exp(xs) / sum(np.exp(xs))

    