###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
# Angad Chandorkar: anajchan
# Roshan Kathawate: rkathawa
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
# a) HOW THE CODE WORkS:
#    The code can be broadly divided into the stipulated tasks: 
#    0. The initial probabilites are computed using the training data. Both the log and normal values of prior, emission and transition
#       probabilites are stored during this stage. Log values are used for computation in the Naive and Viterbi Algorithms.
#    1. Naive Bayes: The naive bayes has been implemented using the probabilities computed in the previous step. Some optimization 
#       is achieved here for words not observed in the training set. This is a small optimization done so as not to blindly
#       classify new words (e.g. if the word has a punctuation, classification is restricted to Noun, Verb and Prt) 
#    2. MCMC: Here, for classifying each word, we sample from the normalized distribution for its posterior probability. For every
#       statement, we take 15 samples and select the last 5. We ran a number of experiments to find an appropriate balance between time
#       required and corresponding accuracy before finalizing this number.   
#    3. Approximate Max Marginal Inference: In this step, we used 100 samples generated using MCMC before computing the probabilities for 
#       each word in the sentence. Again, we ran a number of experiments before arriving at the number of samples. We observed that there   
#       was only a marginal increase in accuracy if we increased the number of samples beyond 1000.
#    4. Viterbi: Here, we have created a bi-directional graph strucutre to simulate the trellis. For each word, the posterior probability
#       is generated for the 12 label objects for each word of the sentence. For each of these objects, the score to the best 
#       child node isstored in the object. For each child node, a dictionary of the score from all parent nodes is also maintained. 
#       These values are utilized during the Backtrace to identify the sequence of lables in each of the time-steps
#    5. Best Algorithm: For all of the implemented algorithms, the best results on the training and test data were obtained for the Viterbi
#       Algorithm. We were not able to sufficiently improve upon the accuracy for this algorithm despite a few tweaks and are therefore 
#       including the same as the Best Algorithm.  
#
# b) RESULTS OF EVALUATION ON BC.TEST
#    ALGORITHM                            WORDS CORRECT (%)                  CORRECT SENTENCES (%)         
#    1. Naive                                 92.82                                 42.25
#    2. Sampling MCMC                         89.25                                  7.55
#    3. Max Marginal using MCMC               90.21                                  8.40
#    4. Viterbi                               93.40                                 45.70
#    5. Best(Viterbi)                         93.40                                 45.70
#
#
# c) PROBLEMS FACED, ASSUMPTIONS, DESIGN DECISIONS
#    Implemnting Smoothing was a problem. We tried to use Add One smoothing but weren't able to decide on what the vocabulary
#    should be for different conditional probability computations. Therefore, we have made the assumption that if the probability 
#    of an event is 0, we substitute this with 0.000000001. We checked for any drops in accuracy before arriving at this value. 
#    For ensuring the accuracy we used both bc.test.tiny and bc.test files.  
#    Also, while both MCMC and Viterbi are yielding proper results, both algorithms could have been implemented in a much more concise manner  
#    
#
####

from __future__ import division
import random
import math
import operator
import itertools
import sys
import string
    
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
strt_p = {}    
strt_p2 = {}
trans_p = {}
trans_p2  = {}   
emit_p  = {}    
emit_p2  = {}
tag_p = {}      
globaltags = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.'] 


class Node:

    def __init__(self, key, word):
        self.d_key = key
        self.d_sndx = word 
        self.d_val = None
        self.t_chld = {'adj':0.0, 'adv':0.0, 'adp':0.0, 'conj':0.0, 'det':0.0, 'noun':0.0, 'num':0.0, 'pron':0.0, 'prt':0.0, 'verb':0.0, 'x':0.0, '.':0.0}
        self.f_prnt = {'adj':0.0, 'adv':0.0, 'adp':0.0, 'conj':0.0, 'det':0.0, 'noun':0.0, 'num':0.0, 'pron':0.0, 'prt':0.0, 'verb':0.0, 'x':0.0, '.':0.0}
        self.bst_chld = {}
        self.bst_prnt = {}

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        global emit_p
        global strt_p
        global trans_p
        k = len(sentence)
        p_sum = 0
        
        for word in range(k):
           
            emission_c = sentence[word] + ' | ' +label[word] 
            if emission_c not in emit_p:
                emission_p = math.log(0.000000001,10)
            else:     
                emission_p = emit_p[emission_c]

            if word == 0:   
               p_sum = p_sum + strt_p[label[word]] + emission_p
            else:
                p_sum = p_sum + emission_p + trans_p[label[word] + ' | ' +label[word-1]]
        return p_sum

    # Do the training!
    def train(self, data):
          
        global strt_p     
        global trans_p
        global trans_p2   
        global emit_p
        global emit_p2    
        global tag_p
        global globaltags 
        global norm_strt_p
        sentnum = 0
        trans_c = {}
        emit_c = {}
        tag_c = {}

        for tag in globaltags:
            strt_p[tag] = 0
            tag_c[tag] = 0

        trans_lst = list(itertools.permutations(globaltags, 2))
        for tag_perms in trans_lst:
            trans_cond = tag_perms[0] + ' | ' + tag_perms[1]
            trans_cnd2 = tag_perms[0] + ' | ' + tag_perms[0]
            trans_c[trans_cond] = 0
            if trans_cnd2 not in trans_c: trans_c[trans_cnd2] = 0 
        
        for i,j in data:
            sentnum += 1    
            
            ##TAG_C - COUNT OF ALL TAGS
            for tag in j:
                if tag not in tag_c.keys(): tag_c[tag] = 1
                else: tag_c[tag] = tag_c[tag]+1     
            
            ##C(S1) - STARTING POS
            if j[0] not in strt_p.keys(): strt_p[j[0]] = 1    
            else: strt_p[j[0]] += 1
            
            ##TRANSITION 
            for z in range(len(j)):
                if z>0:
                    k = j[z] + ' | ' + j[z-1]
                    if k not in trans_c: trans_c[k] = 1
                    else: trans_c[k] = trans_c[k] + 1
            
            ##EMISSION
            for z in range(len(i)):
                k = i[z] + ' | ' + j[z] 
                if k not in emit_c: 
                    emit_c[k] = 1
                else: emit_c[k] = emit_c[k]+1
       
        ##P(S1)         
 
        for key in strt_p: 
            if strt_p[key] == 0:
                strt_p2[key] = 0.000000001
                strt_p[key] = math.log(0.000000001, 10)             
            else:
                strt_p2[key] = strt_p[key]/sentnum
                strt_p[key] = math.log(strt_p[key]/sentnum, 10)            

      ##P(Ti+1|Ti)
        for key in trans_c.keys():
            conditionedover = key.split(' | ')[1]
            if tag_c[conditionedover] != 0:
                if trans_c[key] != 0:
                    trans_p[key] = math.log((trans_c[key]/tag_c[conditionedover]), 10)
                    trans_p2[key] = trans_c[key]/tag_c[conditionedover]
                else:
                    trans_p[key] = math.log(0.000000001, 10)
                    trans_p2[key] =  0.000000001   
            else:
               trans_p[key] = math.log(0.000000001, 10)
        
        ##P(Wi|Tj)          
        for key in emit_c.keys():
            conditionedover = key.split(' | ')[1]
            if tag_c[conditionedover] != 0:
                if emit_c[key] != 0:
                    emit_p[key] = math.log(emit_c[key]/tag_c[conditionedover], 10)
                    emit_p2[key] = emit_c[key]/tag_c[conditionedover]
                else:
                    emit_p[key] = math.log(0.000000001, 10)
                    emit_p2[key] = 0.000000001
            else: 
                emit_p[key] = math.log(0.000000001,10)
       
        ##P(T)
        tagsamplespace = sum(tag_c.values())
        for key in tag_c:
            if tag_c[key] != 0:
                tag_p[key] = math.log(tag_c[key]/tagsamplespace)  
            else:
                tag_p[key] = math.log(0.000000001)    
    
       
    # Functions for each algorithm.
    #
   
    def naive(self, sentence):
        
        global emit_p 
        global tag_p
        ret_lst = []

        for word in sentence:
            temp_dict = {}
            wordfound = False    
           
            for tag in globaltags:
                find_emit = word + ' | ' + tag
                
                if find_emit in emit_p:
                    wordfound = True
                    temp_dict[find_emit] = emit_p[find_emit]

            if wordfound == False:
                if word.isalpha() == False:
                    if word.isdigit():
                        ##NUMBER
                        find_emit = word + ' | ' + globaltags[6]                       
                    elif len(word) == 1:
                        for c in string.punctuation:
                            if c == word:
                                ##PUNCT
                                find_emit = word + ' | ' + globaltags[11]                                 
                    elif ((len(word) >= 2) and ((word[-2:] == "'t") or (word[-3:] == "n't"))):
                        ##PUNCT+WORDS&VERB
                        find_emit = word + ' | ' +globaltags[9]
                        
                    elif ((len(word) >= 2) and ((word[-2:] == "'d") or (word[-3:] == "'ve"))):
                        ##PUNCT+WORDS&PRT
                        find_emit = word + ' | ' +globaltags[8]
                       
                    else:
                        ##PUNCT+WORDS&NOUN
                        find_emit = word + ' | ' +globaltags[5]

                else:
                    ##RANDOM WORD&VERB
                    if (len(word) >= 4) and (word[-2:] == 'ed' or  word[-3:] == 'ing'):
                        find_emit = word + ' | ' +globaltags[9]  
                    ##RANDOM WORD&NOUN
                    else:       
                        find_emit = word + ' | ' +globaltags[5]
                      
                temp_dict[find_emit] = math.log(0.000000001,10)


            for key in temp_dict:
                tag = key.split(' | ')[1]
                temp_dict[key] = temp_dict[key] + tag_p[tag]
                
            argmax_key = max(temp_dict.iteritems(), key=operator.itemgetter(1))[0]
            ret_lst.append(argmax_key.split(' | ')[1]) 

        return [ [ret_lst], [] ]

   
    def mcmc(self, sentence, samplemit_c):

        global emit_p 
        global tag_p
        global emit_p2
        global trans_p2
        global norm_strt_p
        global strt_p2
        smpl_lst = []
        smpl_lst_tag = []
        smpl_arch = {}
        ret_lst = []
        smpl_count = 0
        smpl_dct = {}
        x_lst = ''

        ret_lst = [[],[],[],[],[]]

        for words in sentence:
            smpl_lst.append(words)
            smpl_lst_tag.append('noun')
            
        l = len(smpl_lst)
        rnge = 15

        for samples in range(rnge):
            
            for word in range(l):           

                if word == 0:
                    start_prob = []
                    x = random.randrange(1,100)
                    wordfound = False
                    
                    #EMITPROB
                    for tags in globaltags:
                        e_indx= smpl_lst[word] + ' | ' + tags
                        if e_indx in emit_p2:
                            emission = emit_p2[e_indx]
                            wordfound = True
                        else:
                            emission = 0.000000001

                        #STARTPROB
                        if tags in strt_p:
                            strtprob = strt_p2[tags]
                        else:
                            strtprob = 0.000000001

                        start_prob.append([[tags],[emission * strtprob]])

                    if 1:

                        sum_val = 0
                        for itm in start_prob:
                            sum_val  = sum_val + itm[1][0]
     
                        for itm in start_prob:
                            itm[1][0] = (itm[1][0]/sum_val)*100

                        start_prob = sorted(start_prob, key = operator.itemgetter(1), reverse=True)
                        temp = 0
                        for i in start_prob:
                            temp = i[1][0] + temp
                            i[1] = temp

                        rndm = random.randrange(1,100)
                        for itm in start_prob:
                            if rndm - itm[1] <= 0:
                                smpl_lst_tag[word] == start_prob[0][0]     
               
                if word == l-1:
                    sorted_tags = []
                    t_prv = smpl_lst_tag[word-1]  
                    wordfound = False                 

                    s = 0
                    for tags in globaltags:
                    #Emission prob for all twelve tags:
                        e_indx= smpl_lst[word] + ' | ' + tags
                        if e_indx in emit_p2:
                            emission = emit_p2[e_indx]
                            wordfound = True
                        else:
                            emission = 0.000000001
                    #Transition1:
                        tp_indx = tags + ' | ' + t_prv
                        if tp_indx in trans_p2:
                            transition_p = trans_p2[tp_indx] 
                        else:
                            transition_p = 0.000000001
                        sorted_tags.append([[tags],[emission * transition_p]])


                    if 1:
                        sm = 0
                        for items in sorted_tags: 
                            sm = sm + items[1][0]
                        
                        for items in sorted_tags: 
                            items[1] = (items[1][0]/sm)*100
                                                                              
                        sorted_tags = sorted(sorted_tags, key = operator.itemgetter(1), reverse=True)
                        
                        for i in range(len(sorted_tags)):
                            if i > 0:
                                sorted_tags[i][1] = sorted_tags[i][1] + sorted_tags[i-1][1]
                    
                        rndm = random.randrange(1,100)

                        for i in range(len(sorted_tags)):
                            if rndm - sorted_tags[i][1] <= 0:
                                smpl_lst_tag[word] = sorted_tags[i][0][0]
                        
                                if samples >= rnge-5:
                                    for items in smpl_lst_tag:
                                        x_lst = x_lst + items + '&*&' 
                                    x_lst = x_lst + '|||'
                                break

                if (word != 0) and (word != l-1) :
                    #"INTERMEDIATE WORD"
                    sorted_tags = []
                    t_prv = smpl_lst_tag[word-1]
                    t_nxt = smpl_lst_tag[word+1]
                    wordfound = False
                    
                    s = 0
                    for tags in globaltags:
                        #Emission prob for all twelve tags: 
                        e_indx= smpl_lst[word] + ' | ' + tags
                        if e_indx in emit_p2:
                            emission = emit_p2[e_indx]
                            wordfound = True
                        else:
                            emission = 0.000000001

                    #Transition1:
                        tp_indx = tags + ' | ' + t_prv
                        if tp_indx in trans_p2:
                            transition_p = trans_p2[tp_indx] 
                        else:
                            transition_p = 0.000000001
                    #Transition2:
                        tn_indx = t_nxt + ' | ' + tags
                        if tn_indx in trans_p2:
                            transition_n = trans_p2[tn_indx]
                        else:
                            transition_n = 0.000000001
                        sorted_tags.append([[tags],[emission * transition_p * transition_n]])

                    if wordfound == True:

                        sm = 0
                        for items in sorted_tags: 
                            sm = sm + items[1][0]
                        
                        for items in sorted_tags: 
                            items[1] = (items[1][0]/sm)*100
                        
                        sorted_tags = sorted(sorted_tags, key = operator.itemgetter(1), reverse=True)
                        
                        for i in range(len(sorted_tags)):
                            if i > 0:
                                sorted_tags[i][1] = sorted_tags[i][1] + sorted_tags[i-1][1]
                       
                         
                        rndm = random.randrange(1,100)

                        for i in range(len(sorted_tags)):
                            if rndm - sorted_tags[i][1] < 0:
                                smpl_lst_tag[word] = sorted_tags[i][0][0]
                                break

                    elif wordfound == False:
                        if smpl_lst[word].isalpha() == False:
                            if smpl_lst[word].isdigit():
                                ##NUMBER
                                smpl_lst_tag[word] = globaltags[6]                       
                            elif len(smpl_lst[word]) == 1:
                                for c in string.punctuation:
                                    if c == smpl_lst[word]:
                                        ##PUNCT
                                        smpl_lst_tag[word] =  globaltags[11]                                 
                            elif ((len(smpl_lst[word]) >= 2) and ((smpl_lst[word][-2:] == "'t") or (smpl_lst[word][-3:] == "n't"))):
                                ##PUNCT+WORDS&VERB
                                smpl_lst_tag[word] =  globaltags[9]
                                
                            elif ((len(smpl_lst[word]) >= 2) and ((smpl_lst[word][-2:] == "'d") or (smpl_lst[word][-3:] == "'ve"))):
                                ##PUNCT+WORDS&PRT
                                smpl_lst_tag[word] = globaltags[8]
                               
                            else:
                                ##PUNCT+WORDS&NOUN
                               smpl_lst_tag[word] = globaltags[5]

                        else:
                            ##RANDOM WORD&VERB
                            if (len(smpl_lst[word]) >= 4) and (smpl_lst[word][-2:] == 'ed' or  smpl_lst[word][-3:] == 'ing'):
                                smpl_lst_tag[word] = globaltags[9]  
                            ##RANDOM WORD&NOUN
                            else:       
                                smpl_lst_tag[word] = globaltags[5]
 
        y_lst = x_lst.split('|||')
        y_lst.remove('')
        for itm in range(len(y_lst)):
            ret_lst[itm] = y_lst[itm].split('&*&')
            ret_lst[itm].remove('')
                 
        return [ ret_lst, [] ]
    
    def best(self, sentence):
        
        global trans_p
        global emit_p
        global strt_p
        global globaltags
        smpl_lst = []
        ret_lst = []
        v_node = {}

        for word in range(len(sentence)):
            lst = [] 
            smpl_lst.insert(word, sentence[word])
            for tag in globaltags:
                lst.append(Node(tag, word))
            v_node[word] = lst
        l = len(smpl_lst)

        #COMPUTATION

        for word in range(l):
             
            #FOR FIRST WORD
            if word == 0:

                w1 = smpl_lst[word]
                for tag in range(len(globaltags)):

                    #1. COMPUTE FOR WORD 1 
                    c_tag = globaltags[tag]
                    #EMISSION
                    e_w1 = w1 + ' | ' + c_tag
                    if e_w1 in emit_p:
                        pe_w1 = emit_p[e_w1]
                    else: 
                        pe_w1 = math.log(0.000000001,10) 
                    #TAG_STRT
                    ps_t1 = strt_p[c_tag]
                    #STORE VAL
                    v_node[word][tag].d_val = ps_t1 + pe_w1

                    #2. COMPUTE PROBABILITIES FOR GOING TO WORD 2, EMISSION AND PRIORS EXIST, GET TRANSITION PROBABILITIES
                    if (word + 1) < l:
                        w2 = smpl_lst[word+1]
                        chld_dct = v_node[word][tag].t_chld
                        
                        for tag2 in globaltags:
                            e_w2 = w2 + ' | ' + tag2
                            n_tag = tag2 + ' | ' + c_tag
                            if e_w2 in emit_p:
                                pe_w2 = emit_p[e_w2]
                            else: 
                                pe_w2 = math.log(0.000000001,10) 
                            
                            ps_t2 = trans_p[n_tag]
                            
                            chld_dct[tag2] = v_node[word][tag].d_val + pe_w2 +  ps_t2
                        min_cst_chld = max(chld_dct.items(), key=lambda x: x[1])
                        chld_dct = {}
                        v_node[word][tag].bst_chld[min_cst_chld[0]] = min_cst_chld[1]
                        nxt_tag = min_cst_chld[0] 
                        nxt_indx = globaltags.index(nxt_tag)
                        v_node[word+1][nxt_indx].bst_prnt[c_tag] = min_cst_chld[1]
                                 
        #FOR INTERMEDIATE WORDS
            if ((word != 0) and (word < l-1)):

                w1 = smpl_lst[word]
                w2 = smpl_lst[word+1]

                
                for tag in range(len(globaltags)):
                    c_tag = globaltags[tag]
                    
                    if v_node[word][tag].bst_prnt:
                        prior = max(v_node[word][tag].bst_prnt.iteritems(), key=operator.itemgetter(1))[1]
                      
                    
                    #FOR SELECTED TAG OBJECT IN THE TIME STEP, COMPUTE THE EMISSION+TRANSITION+PRIOR FOR THE 12 TAGS FROM NEXT TIME STEP

                        for ntag in range(len(globaltags)):
                            tag2 = globaltags[ntag] 
                            chld_dct = v_node[word][tag].t_chld
                            
                            #EMISSION for w2 for n_tag
                            e_w2 = w2 + ' | ' + tag2
                            if e_w2 in emit_p:
                                pe_w2 = emit_p[e_w2]
                            else: 
                                pe_w2 = math.log(0.000000001,10) 
                            #TRANSITION TO N_TAG GIVEN C_TAG
                            n_tag = tag2 + ' | ' + c_tag
                            ps_t2 = trans_p[n_tag]                               
                            chld_dct[tag2] = prior + ps_t2+ pe_w2   
                        min_cst_chld = max(chld_dct.items(), key=lambda x: x[1])
                        chld_dct = {}
                        v_node[word][tag].bst_chld[min_cst_chld[0]] = min_cst_chld[1]
                        nxt_tag = min_cst_chld[0] 
                        nxt_indx = globaltags.index(nxt_tag)
                        v_node[word+1][nxt_indx].bst_prnt[c_tag] = min_cst_chld[1]       

        #BACKTRACE         
        bck_dct = {} 
        for word in range(l):
            back = l-word-1
            if l > 1: 
                if back != 0:
                    
                    for itm in v_node[back]:
                        if itm.bst_prnt:
                            bst_prnt = max(itm.bst_prnt.iteritems(), key=operator.itemgetter(1))[0]
                            bst_prnt_cst = max(itm.bst_prnt.iteritems(), key=operator.itemgetter(1))[1]
                            bck_dct[itm.d_key + ' | ' + bst_prnt] = [bst_prnt_cst]
                        else: 
                            pass    

                    p_key = max(bck_dct.iteritems(), key=operator.itemgetter(1))[0]    
                    ret_lst.append(p_key.split(' | ')[0])
                   
                    if back == 1: 
                        ret_lst.append(p_key.split(' | ')[1])
            if l == 1:
                
                one_dct = {}
                for itm in v_node[0]:
                    one_dct[itm.d_key] = itm.d_val
                one_key = max(one_dct.iteritems(), key=operator.itemgetter(1))[0]
                ret_lst.append(one_key)


        ret_lst.reverse()
        return [ [ret_lst], [] ]
    
  
    def max_marginal(self, sentence):

        global emit_p 
        global tag_p
        global emit_p2
        global trans_p2
        global norm_strt_p
        global strt_p2
        smpl_lst = []
        smpl_lst_tag = []
        smpl_arch = {}
        ret_lst = []
        smpl_count = 0
        smpl_dct = {}
        x_lst = ''
        tag_dct_c = {}
        tag_dct = {}
        prb_lst = []
        ret_lst = []

        for words in sentence:
            smpl_lst.append(words)
            smpl_lst_tag.append('noun')
            
        for tag in globaltags:
            tag_dct[tag] = 0

        for count in range(len(smpl_lst)):
            tag_dct_c[count] = {}
            for tag in globaltags:
                tag_dct_c[count][tag] = 0

   
        l = len(smpl_lst)
        rnge = 100

        for samples in range(rnge):
            
            for word in range(l):           

                if word == 0:

                    start_prob = []
                    x = random.randrange(1,100)
                    wordfound = False
                    
                    #EMITPROB
                    for tags in globaltags:
                      
                        e_indx= smpl_lst[word] + ' | ' + tags
                        if e_indx in emit_p2:
                            emission = emit_p2[e_indx]
                            wordfound = True
                        else:
                            emission = 0.000000001

                        #STARTPROB
                        if tags in strt_p:
                            strtprob = strt_p2[tags]
                        else:
                            strtprob = 0.000000001

                        start_prob.append([[tags],[emission * strtprob]])

                    if 1:

                        sum_val = 0
                        for itm in start_prob:
                            sum_val  = sum_val + itm[1][0]
     
                        for itm in start_prob:
                            itm[1][0] = (itm[1][0]/sum_val)*100    

                        start_prob = sorted(start_prob, key = operator.itemgetter(1), reverse=True)
                        temp = 0
                        for i in start_prob:
                            temp = i[1][0] + temp
                            i[1] = temp

                        rndm = random.randrange(1,100)
                        for itm in start_prob:
                            if rndm - itm[1] <= 0:
                                smpl_lst_tag[word] == start_prob[0][0]     
                  
                
                if word == l-1:
                    sorted_tags = []
                    t_prv = smpl_lst_tag[word-1]  
                    wordfound = False                 

                    s = 0
                    for tags in globaltags:
                    #Emission prob for all twelve tags:
                        e_indx= smpl_lst[word] + ' | ' + tags
                        if e_indx in emit_p2:
                            emission = emit_p2[e_indx]
                            wordfound = True
                        else:
                            emission = 0.000000001
                    #Transition1:
                        tp_indx = tags + ' | ' + t_prv
                        if tp_indx in trans_p2:
                            transition_p = trans_p2[tp_indx] 
                        else:
                            transition_p = 0.000000001

                        sorted_tags.append([[tags],[emission * transition_p]])

                    if 1:

                        sm = 0
                        for items in sorted_tags: 
                            sm = sm + items[1][0]
                        
                        for items in sorted_tags: 
                            items[1] = (items[1][0]/sm)*100

                                                                                
                        sorted_tags = sorted(sorted_tags, key = operator.itemgetter(1), reverse=True)
                        
                        for i in range(len(sorted_tags)):
                            if i > 0:
                                sorted_tags[i][1] = sorted_tags[i][1] + sorted_tags[i-1][1]    
                        
                        rndm = random.randrange(1,100)

                        for i in range(len(sorted_tags)):
                            if rndm - sorted_tags[i][1] <= 0:
                                smpl_lst_tag[word] = sorted_tags[i][0][0]
                        
                                for count in range(l):
                                    tag = smpl_lst_tag[count]
                                    tag_dct_c[count][tag] += 1

                                break 


                if (word != 0) and (word != l-1) :
                    
                    sorted_tags = []
                    t_prv = smpl_lst_tag[word-1]
                    t_nxt = smpl_lst_tag[word+1]
                   
                    wordfound = False
                    
                    s = 0
                    for tags in globaltags:
                        #Emission prob for all twelve tags: 
                        e_indx= smpl_lst[word] + ' | ' + tags
                        if e_indx in emit_p2:
                            emission = emit_p2[e_indx]
                            wordfound = True
                        else:
                            emission = 0.000000001
                    #Transition1:
                        tp_indx = tags + ' | ' + t_prv
                        if tp_indx in trans_p2:
                            transition_p = trans_p2[tp_indx] 
                        else:
                            transition_p = 0.000000001
                    #Transition2:
                        tn_indx = t_nxt + ' | ' + tags
                        if tn_indx in trans_p2:
                            transition_n = trans_p2[tn_indx]
                        else:
                            transition_n = 0.000000001

                        sorted_tags.append([[tags],[emission * transition_p * transition_n]])

                    if wordfound == True:

                        sm = 0
                        for items in sorted_tags: 
                            sm = sm + items[1][0]
                        
                        for items in sorted_tags: 
                            items[1] = (items[1][0]/sm)*100 

                        sorted_tags = sorted(sorted_tags, key = operator.itemgetter(1), reverse=True)
                        
                        for i in range(len(sorted_tags)):
                            if i > 0:
                                sorted_tags[i][1] = sorted_tags[i][1] + sorted_tags[i-1][1]
                       
                        rndm = random.randrange(1,100)

                        for i in range(len(sorted_tags)):
                            if rndm - sorted_tags[i][1] < 0:
                                smpl_lst_tag[word] = sorted_tags[i][0][0]
                                break

                    elif wordfound == False:
                        if smpl_lst[word].isalpha() == False:
                            if smpl_lst[word].isdigit():
                                ##NUMBER
                                smpl_lst_tag[word] = globaltags[6]                       
                            elif len(smpl_lst[word]) == 1:
                                for c in string.punctuation:
                                    if c == smpl_lst[word]:
                                        ##PUNCT
                                        smpl_lst_tag[word] =  globaltags[11]                                 
                            elif ((len(smpl_lst[word]) >= 2) and ((smpl_lst[word][-2:] == "'t") or (smpl_lst[word][-3:] == "n't"))):
                                ##PUNCT+WORDS&VERB
                                smpl_lst_tag[word] =  globaltags[9]
                                
                            elif ((len(smpl_lst[word]) >= 2) and ((smpl_lst[word][-2:] == "'d") or (smpl_lst[word][-3:] == "'ve"))):
                                ##PUNCT+WORDS&PRT
                                smpl_lst_tag[word] = globaltags[8]
                               
                            else:
                                ##PUNCT+WORDS&NOUN
                               smpl_lst_tag[word] = globaltags[5]

                        else:
                            ##RANDOM WORD&VERB
                            if (len(smpl_lst[word]) >= 4) and (smpl_lst[word][-2:] == 'ed' or  smpl_lst[word][-3:] == 'ing'):
                                smpl_lst_tag[word] = globaltags[9]  
                            ##RANDOM WORD&NOUN
                            else:       
                                smpl_lst_tag[word] = globaltags[5]
                            # print "FOR I WORD: ", smpl_lst[word], " UPDATED TAG TO: ", smpl_lst_tag[word]
                
        for count in range(l):
            tmp_dct = tag_dct_c[count]
            max_key = max(tmp_dct.iteritems(), key=operator.itemgetter(1))[0]
            key_prb = tag_dct_c[count][max_key]/rnge
            ret_lst.insert(count, max_key)     
            prb_lst.insert(count, key_prb)

        return [ [ret_lst], [prb_lst,] ]
        

   
    def viterbi(self, sentence):

        global trans_p
        global emit_p
        global strt_p
        global globaltags
        smpl_lst = []
        ret_lst = []
        v_node = {}

        for word in range(len(sentence)):
            lst = [] 
            smpl_lst.insert(word, sentence[word])
            for tag in globaltags:
                lst.append(Node(tag, word))
            v_node[word] = lst
        l = len(smpl_lst)

        #COMPUTATION

        for word in range(l):
             
            #FOR FIRST WORD
            if word == 0:

                w1 = smpl_lst[word]

                for tag in range(len(globaltags)):

                    #1. COMPUTE FOR WORD 1 
                    c_tag = globaltags[tag]

                    #EMISSION
                    e_w1 = w1 + ' | ' + c_tag
                    if e_w1 in emit_p:
                        pe_w1 = emit_p[e_w1]
                    else: 
                        pe_w1 = math.log(0.000000001,10) 
                    #TAG_STRT
                    ps_t1 = strt_p[c_tag]
                    #STORE VAL
                    v_node[word][tag].d_val = ps_t1 + pe_w1

                    #2. COMPUTE PROBABILITIES FOR GOING TO WORD 2, EMISSION AND PRIORS EXIST, GET TRANSITION PROBABILITIES
                    if (word + 1) < l:
                        w2 = smpl_lst[word+1]
                        chld_dct = v_node[word][tag].t_chld
                        
                        for tag2 in globaltags:
                            e_w2 = w2 + ' | ' + tag2
                            n_tag = tag2 + ' | ' + c_tag
                            if e_w2 in emit_p:
                                pe_w2 = emit_p[e_w2]
                            else: 
                                pe_w2 = math.log(0.000000001,10) 
                            
                            ps_t2 = trans_p[n_tag]
                            
                            chld_dct[tag2] = v_node[word][tag].d_val + pe_w2 +  ps_t2
                        min_cst_chld = max(chld_dct.items(), key=lambda x: x[1])
                        chld_dct = {}
                        v_node[word][tag].bst_chld[min_cst_chld[0]] = min_cst_chld[1]
                        nxt_tag = min_cst_chld[0] 
                        nxt_indx = globaltags.index(nxt_tag)
                        v_node[word+1][nxt_indx].bst_prnt[c_tag] = min_cst_chld[1]
                                 
        #FOR INTERMEDIATE WORDS
            if ((word != 0) and (word < l-1)):

                w1 = smpl_lst[word]
                w2 = smpl_lst[word+1]
                
                for tag in range(len(globaltags)):
                    c_tag = globaltags[tag]
                    
                    if v_node[word][tag].bst_prnt:
                        prior = max(v_node[word][tag].bst_prnt.iteritems(), key=operator.itemgetter(1))[1]
                                          
                    #FOR SELECTED TAG OBJECT IN THE TIME STEP, COMPUTE THE EMISSION+TRANSITION+PRIOR FOR THE 12 TAGS FROM NEXT TIME STEP

                        for ntag in range(len(globaltags)):
                            tag2 = globaltags[ntag] 
                            chld_dct = v_node[word][tag].t_chld
                           
                            #EMISSION for w2 for n_tag
                            e_w2 = w2 + ' | ' + tag2
                            if e_w2 in emit_p:
                                pe_w2 = emit_p[e_w2]
                            else: 
                                pe_w2 = math.log(0.000000001,10) 
                            #TRANSITION TO N_TAG GIVEN C_TAG
                            n_tag = tag2 + ' | ' + c_tag
                            ps_t2 = trans_p[n_tag]                               
                            chld_dct[tag2] = prior + ps_t2+ pe_w2   

                        min_cst_chld = max(chld_dct.items(), key=lambda x: x[1])
                        chld_dct = {}
                        v_node[word][tag].bst_chld[min_cst_chld[0]] = min_cst_chld[1]
                        nxt_tag = min_cst_chld[0] 
                        nxt_indx = globaltags.index(nxt_tag)
                        v_node[word+1][nxt_indx].bst_prnt[c_tag] = min_cst_chld[1]       

        #BACKTRACE         
        bck_dct = {} 
        for word in range(l):
            back = l-word-1
            if l > 1: 
                if back != 0:                   
                    for itm in v_node[back]:

                        if itm.bst_prnt:
                            bst_prnt = max(itm.bst_prnt.iteritems(), key=operator.itemgetter(1))[0]
                            bst_prnt_cst = max(itm.bst_prnt.iteritems(), key=operator.itemgetter(1))[1]
                            bck_dct[itm.d_key + ' | ' + bst_prnt] = [bst_prnt_cst]
                        else: 
                            pass    

                    p_key = max(bck_dct.iteritems(), key=operator.itemgetter(1))[0]    
                    ret_lst.append(p_key.split(' | ')[0])
                    
                    if back == 1: 
                        ret_lst.append(p_key.split(' | ')[1])
            if l == 1:
                
                one_dct = {}
                for itm in v_node[0]:
                    one_dct[itm.d_key] = itm.d_val
                one_key = max(one_dct.iteritems(), key=operator.itemgetter(1))[0]
                ret_lst.append(one_key)


        ret_lst.reverse()
        return [ [ret_lst], [] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

