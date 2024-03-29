from re import S
import numpy as np
import multiprocess as mp
from scipy.linalg import null_space
import os
import pickle as pkl


class EA:
    def __init__(self, num_params, log_dir="./log", mu=2, num_select=2, num_offspring=5, num_parent=10, use_multiprocess=False, num_overlap=1, num_process=4, do_mutate=True, crossover_type="pcx"):
        self.num_parent = int(num_parent)
        self.num_offspring = int(num_offspring)
        self.num_params = int(num_params)
        self.pmin = None
        self.pmax = None
        self.log_dir = log_dir
        self.use_multiprocess = use_multiprocess
        self.job_id = 0
        self.clock = 0
        self.num_process = num_process
        self.num_select = num_select
        self.num_overlap = num_overlap

        self.mu = mu
        self.param_vec = np.zeros([self.num_params, self.num_parent])
        self.fit_score = np.zeros([self.num_parent])
        self.fobj = None
        self.parent_id = np.ones(self.num_parent) * (-1)
        self.offspring_id = []
        # A Functional Specialization Hypothesis for Designing Genetic Algorithms 
        self.sgm_eta = 1/np.sqrt(self.mu)
        self.sgm_xi  = 0.35/np.sqrt(self.num_parent - self.mu)
        self.do_mutate = do_mutate
        self.crossover_type = crossover_type

    def set_object_func(self, f):
        # object function need to return float (fitness)
        # res = f([arr, job_id])
        self.fobj = f 

    def set_min_max(self, pmin, pmax):
        if (len(pmin) != self.num_params) or (len(pmin) != self.num_params):
            print("The number of min-max is wrong")
            return

        self.pmin = np.array(pmin)
        self.pmax = np.array(pmax)

    def check_setting(self):
        if self.num_parent < self.num_offspring:
            raise ValueError("The number of offspring is larger than # parents")

        if self.num_select > self.num_offspring:
            raise ValueError("Too large num_select")

        if self.mu >= self.num_params:
            raise ValueError("mu cannot be larger than # params")

        if self.pmin is None or self.pmax is None:
            raise AttributeError("Boundary is not defined: call set_min_max")

        if self.fobj is None:
            raise AttributeError("Object function is not defined: call set_object_func")

        if self.crossover_type not in ["pcx", "undx"]:
            raise ValueError("Crossover type need need to be pcx or undx, selected %s"%(self.crossover_type))

    def run(self, max_iter=100, tol=1e-3, auto_init=True):
        job_id = 0
        ncycle = 0
        dscore = 0

        # check setting
        self.check_setting()

        if auto_init:
            self.random_initialization()

        for n in range(int(max_iter)):
            self.next_generation()
            self.print_log()

    def random_initialization(self):
        # all parent have id as -1
        for n in range(self.num_parent):
            self.param_vec[:, n] = np.random.uniform(self.pmin, self.pmax)
        self.eval_initialization()

    def eval_initialization(self):
        if self.use_multiprocess:
            args = []
            for n in range(self.num_parent):
                args.append([self.param_vec[:,n], self.job_id])
                self.count_job()

            self.fit_score = []
            flag = False
            n_set = self.num_parent // self.num_process
            if self.num_parent % self.num_process != 0:
                flag = True
                n_set += 1

            for n in range(n_set):
                n0 = self.num_process * n
                with mp.Pool(self.num_process) as p:
                    if flag and (n == n_set-1):
                        args_tmp = args[n0:]
                    else:
                        args_tmp = args[n0:n0+self.num_process]

                    tmp_fit = p.map(self.fobj, args_tmp)
                    self.fit_score.extend(tmp_fit)
        else:
            self.fit_score = []
            for n in range(self.num_parent):
                self.fit_score.append(self.fobj([self.param_vec[:,n], self.job_id]))
                self.count_job()
        
        self.fit_score = np.array(self.fit_score)
        self.reset_job_id()

    def count_job(self):
        if (len(self.offspring_id) == self.num_offspring):
            self.offspring_id = []
        self.offspring_id.append(self.job_id)
        self.job_id += 1

    def reset_job_id(self):
        self.job_id = 0
        self.offspring_id = []

    def next_generation(self):
        # get offsprings & evaluate scores
        offspring = self.crossover()
        if self.do_mutate:
            offspring = self.mutate(offspring)

        if self.use_multiprocess:
            # split data
            args = []
            for n in range(self.num_offspring):
                args.append([offspring[:,n], self.job_id])
                self.count_job()

            fitness = []
            for n in range(self.num_overlap):
                with mp.Pool(self.num_process) as p:
                    fitness_tmp = p.map(self.fobj, args[n*self.num_process:(n+1)*self.num_process])
                fitness.extend(fitness_tmp)

        else:
            fitness = []
            for n in range(self.num_offspring):
                fitness.append(self.fobj([offspring[:,n], self.job_id]))
                self.count_job()

        # select parent to change
        id_selected, _ = self.pick_id(self.num_parent, self.num_select)

        # evalutate score
        pop_scores = np.zeros(self.num_offspring+self.num_select)
        pop_scores[:-self.num_select] = fitness
        pop_scores[-self.num_select:] = self.fit_score[id_selected]
        buffer_id = self.offspring_id.copy()
        buffer_param = self.param_vec.copy()
        for nid in id_selected:
            # To differentiate with offspring
            # add index not parent job id (because at the initial, all the parent have -1 as job id)
            buffer_id.append(-int(nid)-1) # To include "0"  
        id_selected = list(id_selected)

        # natural selection
        id_live = self.natural_selection(pop_scores) # index
        # print(id_live, pop_scores.shape, len(buffer_id), len(id_selected), len(self.offspring_id), self.num_offspring)

        # print("buffer id:", buffer_id)
        # print("id_live:", id_live)
        # print("id_selected:", id_selected)
        # print("id; ", [buffer_id[n] for n in id_live])

        # check is there any negative index in buffer_id
        # - id_live: index who lived
        # - buffer_id: [job_id for offspring, -id_selected]
        # - id_selected: index for selected parents

        n = 0
        while n < len(id_live):
            nid = id_live[n]
            b_id = buffer_id[nid]
            if b_id < 0:
                id_live.remove(nid)
                buffer_id.pop(nid)
                id_selected.remove(-b_id-1)
                for i in range(n, len(id_live)):
                    id_live[i] -= 1
                n -= 1
            n += 1

        for n in id_live:
            # print("n: %d, len_id: %d"%(n, len(buffer_id)))
            nid = buffer_id[n] # index of the parent
            if nid < 0:
                # print("pop: ", n, nid)
                id_live.pop(n)
                buffer_id.pop(n)
                id_selected.remove(-nid-1)
        # print("id_live:", id_live)

        # change the parent to offspring
        for n in range(len(id_live)):
            nid = id_live[n]
            new_id = id_selected[n]
            # print(nid, new_id)
            self.param_vec[:, new_id] = offspring[:, nid]
            self.fit_score[new_id] = pop_scores[nid]
            self.parent_id[new_id] = buffer_id[nid]


        # for n in range(self.num_select):
        #     nid = id_live[n]
        #     new_id = id_selected[n]
        #     if nid < self.num_offspring:
        #         self.param_vec[:, new_id] = offspring[:, nid]
        #     else:
        #         nold = id_selected[nid - self.num_offspring] 
        #         self.param_vec[:, new_id] = buffer_param[:, nold]

        #     self.fit_score[new_id] = pop_scores[nid]
        #     self.parent_id[new_id] = buffer_id[nid]

        self.clock += 1

    # def natural_selection(self, fitness, num_select=5):
    #     id_sort = np.argsort(fitness)[::-1]
    #     return id_sort[:num_select]

    def natural_selection(self, fitness, num_opt_select=2):
        id_tot = list(range(self.num_offspring+self.num_select))
        # find the best model
        id_select = []
        for n in range(num_opt_select):
            n_best = np.nanargmax(fitness[id_tot])
            id_tot.pop(n_best)
            id_select.append(n_best)

        # clean nan
        id_nan = np.where(np.isnan(fitness))[0]
        id_tot = remove_index(id_tot, id_nan)

        # use Roullete-Wheel method
        fitness_pos = fitness[id_tot]
        fmin = np.nanmin(fitness_pos)
        if fmin < 0:
            fitness_pos += fmin
        fitness_pos = list(fitness_pos)

        for n in range(self.num_select-num_opt_select):
            prob_select = np.array(fitness_pos) / np.sum(fitness_pos)
            p = np.random.rand()
            i, p_cum = 0, 0
            while p_cum < p:
                i += 1
                if (i == len(prob_select)):
                    break

                p_cum += prob_select[i]
            i -= 1
            # pick
            id_select.append(id_tot[i])
            fitness_pos.pop(i)

        id_select = list(np.sort(id_select))

        return id_select

    def crossover(self):
        offspring = np.ones([self.num_params, self.num_offspring]) * (-1)
        for n in range(self.num_offspring):
            # check boundary condition
            flag = True
            stack = 0
            while flag:
                if self.crossover_type == "pcx":
                    offspring_tmp = self.crossover_pcx()
                elif self.crossover_type == "undx":
                    offspring_tmp = self.crossover_undx()

                if all(offspring_tmp <= self.pmax) and all(offspring_tmp >= self.pmin):
                    offspring[:, n] = offspring_tmp
                    break
                
                stack += 1
                if stack == 5:
                    offspring[:, n] = np.random.uniform(self.pmin, self.pmax)
                    break

        return offspring

    def crossover_undx(self):
        # ==================================
        # Ref)
        # H. Kita & M. Yamamura, IEEE, 1999, A Functional Specialization Hypothesis for Designing Genetic Algorithms
        # I. Ono et al., A Real-coded Genetic Algorithm using the Unimodal Normal Distribution Crossover
        # K. Deb et al., A Computationally Efficient Evolutionary Algorithm for Real-Parameter Optimization
        # ==================================

        # select mu parents (mu < n), span the vectorspace V
        id_select, id_remain = self.pick_id(self.num_parent, self.mu)
        g_vec = np.average(self.param_vec[:, id_select], axis=1)
        d_vec = self.param_vec[:, id_select[:-1]] - g_vec[:, np.newaxis]

        # check is d_vec is null vector
        if get_norm(d_vec) < 1e-5:
            return g_vec
        else:
            # find the basis of vector space W that orthogonal to vector space V
            basis = null_space(d_vec.T)

            # find the distance from the vector space which spanned by d_vec
            nd = np.random.choice(id_remain)
            v = self.param_vec[:, nd] - g_vec
            coord = np.array([np.dot(v, basis[:,n]) for n in range(basis.shape[1])])
            D = np.sqrt(np.sum(coord**2))

        # get offspring
        offspring = g_vec
        # 1st term
        eta = np.random.randn(self.mu-1, 1) * self.sgm_eta
        offspring += np.squeeze(np.dot(d_vec, eta))
        # 2nd term
        # xi = np.random.randn(self.num_params-self.mu+1, 1) * self.sgm_xi
        xi = np.random.randn(basis.shape[1], 1) * self.sgm_xi
        offspring += np.squeeze(D * np.dot(basis, xi))
        # print(basis.shape, xi.shape, d_vec.shape, d_vec, self.mu)

        return offspring

    def crossover_pcx(self):
        # select mu parents (mu < n), span the vectorspace V
        id_select, id_remain = self.pick_id(self.num_parent, self.mu)
        g_vec = np.average(self.param_vec[:, id_select], axis=1)

        # select one parents from selected id
        nd = np.random.choice(id_select)
        x_pick = self.param_vec[:, nd]
        d_vec = x_pick - g_vec
        if all(np.array(d_vec) == 0):
            return x_pick

        id_select = list(id_select)
        id_select = remove_element(id_select, nd)

        # calculate average distance
        D = 0
        for i in id_select:
            D += get_distance(self.param_vec[:, i], d_vec)
        D /= self.mu - 1

        # get basis vector which span the perpendicular to vector d
        tmp_vec = np.concatenate([d_vec.reshape([-1, 1]), self.param_vec[:, id_select]], axis=1)
        basis = gram_schmidt(tmp_vec)
        basis = basis[:, 1:]

        # offspring
        offspring = x_pick.copy()
        offspring += np.random.randn() * self.sgm_eta * d_vec
        tmp = D * np.dot(basis, np.random.randn(self.mu-1, 1)) * self.sgm_xi
        offspring += np.squeeze(D * np.dot(basis, np.random.randn(self.mu-1, 1)) * self.sgm_xi)
        
        return offspring
    
    def pick_id(self, max_id, num_pick):
        id_remain = list(range(max_id))
        id_select = np.random.choice(id_remain, num_pick, replace=False)
        for nid in id_select:
            id_remain.remove(nid)
        id_remain = np.array(id_remain)
        return id_select, id_remain

    def mutate(self, offspring):
        p_th = 0.01/self.num_params
        
        for n in range(self.num_offspring):
            for i in range(self.num_params):
                if np.random.rand() < p_th:
                    sgm = (self.pmax[i] - self.pmin[i])/5
                    x = offspring[i, n] + np.random.randn()*sgm
                    if x > self.pmax[i]:
                        x = self.pmax[i]
                    if x < self.pmin[i]:
                        x = self.pmin[i]
                    
                    offspring[i, n] = x

        return offspring

    def print_log(self, skip_save_param=1):
        # save fitness
        with open(os.path.join(self.log_dir, "log.txt"), "a") as fid:
            for n in range(self.num_parent):
                fid.write("%d:%f,"%(self.parent_id[n], self.fit_score[n]))
            fid.write("\n")

        # save parameters
        if self.clock % skip_save_param == 0:
            data = {"job_id": self.parent_id, "params": self.param_vec}
            with open(os.path.join(self.log_dir, "params_%d.pkl"%(self.clock)), "wb") as fid:
                pkl.dump(data, fid)

    def load_history(self, fdir_history=None):
        """
        Load previous evolved data
        """
        import genalg.logger as gl
        import shutil

        if fdir_history is None:
            fdir_history = self.log_dir

        log_obj = gl.Logger(fdir_history)
        max_param_id = len(log_obj.fit_scores)
        log_obj.load_params(max_param_id-1)

        num = np.shape(log_obj.param_set)[0]
        if num != self.num_params:
            raise AttributeError("The # of params in prev (%d) is different withh current mu (%d)"%(num, self.num_params))

        # copy
        if fdir_history != self.log_dir:
            shutil.copy(os.path.join(fdir_history, "log.txt"), 
                        os.path.join(self.log_dir, "log.txt"))
        else: # save original file
            shutil.copy(os.path.join(fdir_history, "log.txt"), 
                        os.path.join(fdir_history, "log_prev.txt"))
        
        self.param_vec = log_obj.param_set
        self.parent_id = log_obj.job_id_set
        self.job_id = max_param_id
        self.clock = max_param_id
        self.fit_score = np.array(log_obj.fit_scores[-1])


# def remove_index(arr_list, id_target):
#     id_target = np.sort(id_target)[::-1]
#     for n in id_target:
#         arr_list.pop(n)
def remove_index(arr_list, id_target):
    return np.delete(arr_list, id_target)
    

def remove_element(arr, target_val):
    arr = list(arr)
    arr.remove(target_val)
    return np.array(arr)


def project(a, b):
    # calculate proj_b(a): project a to b
    sz_b = np.dot(b, b)
    return np.dot(a, b) / sz_b * b


def get_distance(a, b):
    # perpendicular distance to vector b
    l = np.dot(a, b)/norm(b)
    sz = norm(a)
    return np.sqrt(sz**2 - l**2)


def gram_schmidt(arr):
    # arr is the column matrix
    basis = np.zeros_like(arr)
    for n in range(arr.shape[1]):
        sub = 0
        for i in range(n):
            # print(n, n-i-1)
            sub += project(arr[:, n], basis[:, n-i-1])
        basis[:, n] = arr[:, n] - sub
        div = norm(basis[:, n])
        if div != 0:
            basis[:, n] /= div

    return basis


def norm(a):
    return np.sqrt(np.sum(a**2))


def remove_file(fname):
    try:
        os.remove(fname)
    except:
        print(f"{fname} does not exist!")
        

def get_norm(vec):
    return np.sqrt(np.sum(vec**2))


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q