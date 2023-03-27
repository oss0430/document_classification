import torch
import numpy as np
import pandas as pd
import transformers


class Dataset(object):
    ##TODO:
    ## implement dataset
    
    def __init__(
        self,
        path : str = None,
        df : pd.DataFrame = None
    ) :
        if df is None:
            self.df = df
        
        else :
            self.df = pd.read_json(path, lines = True)
        
    
    
    def get_by_index(
        self,
        index : int 
    ):
        return
            

class DataLoader():
    def __init__(
        self,
        dataset : Dataset = None, 
        batch_size : int = 1
    ):
        self.dataset = dataset
        self.len = len(dataset.df) // batch_size
        self.batch_size = batch_size


    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            label (B,): a Variable that stores the target item indices
        """
        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        #print(click_offsets)
        session_idx_arr = self.dataset.session_idx_arr
        #print(session_idx_arr)

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
                
                
class NaiveBayesDataLoader():
    def __init__(
        self,
        dataset : Dataset = None, 
        batch_size : int = 1
    ):
        self.dataset = dataset
        self.len = len(dataset.df) // batch_size
        self.batch_size = batch_size


    def __iter__(self):
        ##TODO:
        ## make iterable input, label
        """ 
        Yields:
            input : string, not tokenized
            label : int
        """
        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]