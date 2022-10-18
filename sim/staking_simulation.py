from posixpath import dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
from glob import glob 
import seaborn as sns
import pdb


class PeerReviewSim():
    def __init__(self):
        print('Preparing Peer Review Simulation... ')

    def setup(self, distribution, n_stakers, paper_acceptance_probability, inflation_per_iteration, n_rounds):
        """
        Set Parameters for the Simulation

        @n_stakers int
        @distribution str, e.g. 'pareto', 'uniform'
        """

        # Simulation Parameters
        self.n_stakers = n_stakers#10
        self.acceptancte_probability = paper_acceptance_probability#0.5 # For papers
        self.inflation_per_iteration = inflation_per_iteration#0 #1 # Absolute
        self.n_rounds = n_rounds
        self.accepted_paper_counter = 0
        self.omitted_paper_counter = 0
        
        # Generate Dict of Stakers with 
        # - uniformly / pareto distributed Stakes
        # - vote status on paper acceptance

        stakes = {}

        if distribution == 'pareto':
            a, m = 3., 20.  # shape and mode
            sample = (np.random.pareto(a, self.n_stakers) + 1) * m

        elif distribution == 'uniform':
            sample = np.random.uniform(0.1, 60., self.n_stakers)
        else:
            raise ValueError('No correct distribution specified for Stakes')

        # Initialize Stake per Staker
        for key in range(self.n_stakers):
            stakes[key] = sample[key]

        # Save initial distribution for later
        self.initial_stakes = stakes

        # Initialize Vote per Staker
        votes = dict.fromkeys(stakes, int(0))
        
        return stakes, votes

    def gini_coefficient(self, x):
        """Compute Gini coefficient of array of values"""
        diffsum = 0
        for i, xi in enumerate(x[:-1], 1):
            diffsum += np.sum(np.abs(xi - x[i:]))
        return diffsum / (len(x)**2 * np.mean(x))

    def vote(self, max_bet_share, balance_losses):
        """
        @max_bet_share, float in [0,1], determines share of stake which can
        be placed in an individual bet. Upper bound of uniform distribution.

        Introduce a punishment term that disincentivises voting in favor of a large imbalanced bet allocation.
        E.g.: deflate to_be_distributed if the sum(loser_stakes)/sum(winner_stakes) is small.
        @balance_losses True | False
        """

        # Draw Bernoulli distribution and weigh with uniformly distributed bet size
        binomial_sample = np.random.binomial(1, self.acceptancte_probability, self.n_stakers)

        acceptance_indiv = []
        for key in self.votes.keys():
            # Determine bet size
            current_stake = self.stakes[key]
            bet_sample = max(np.random.uniform(0, max_bet_share, 1) * current_stake, 0)
            #print('Current bet size vs stake ', bet_sample, ' --- ', current_stake)
            #pdb.set_trace()

            if binomial_sample[key] > 0:
                self.votes[key] = int(1) * bet_sample
            else:
                self.votes[key] = int(-1) * bet_sample

            acceptance_indiv.append(self.votes[key])



        # Find Vote Decision
        acceptance_numerator = sum(acceptance_indiv)

        if acceptance_numerator >= 0:
            #print('Paper Accepted')
            won_vote = int(1)
            lost_vote = int(-1)
            self.accepted_paper_counter += 1
        else:
            #print('Paper Not Accepted')
            won_vote = int(-1)
            lost_vote = int(1)
            self.omitted_paper_counter += 1

        
        # How much has been won / is to be distributed
        winner_stakes = [x for x in acceptance_indiv if np.sign(x) == np.sign(won_vote)]
        loser_stakes = [x for x in acceptance_indiv if np.sign(x) == np.sign(lost_vote)]
        
        sum_loser_stakes = abs(sum(loser_stakes))
        sum_winner_stakes = abs(sum(winner_stakes))
        weight_ratio = sum_loser_stakes / sum_winner_stakes

        if balance_losses:

            # Compensate loser stakes if there was a strong imbalance
            to_be_distributed = min(sum_loser_stakes * weight_ratio, sum_loser_stakes)
        else:
            to_be_distributed = sum_loser_stakes

        # Adjust Stakes according to Bets and Votes
        new_stakes = copy.deepcopy(self.stakes)
        for key in new_stakes.keys():
            #pdb.set_trace()
            if np.sign(self.votes[key]) == np.sign(won_vote):
                #print('Updating Stake: ', self.stakes[key])
                new_stakes[key] += float(abs(acceptance_indiv[key] / sum_winner_stakes) * to_be_distributed)
                new_stakes[key] += self.inflation_per_iteration
                #print('Updated Stake: ', new_stakes[key])
            else:
                new_stakes[key] -= float((abs(acceptance_indiv[key] / sum_loser_stakes) * to_be_distributed))
                new_stakes[key] += self.inflation_per_iteration

        return new_stakes

    def decompose(self, df, colnames = ['fname', 'n_participants']):

        # Retrieve n_participants from filename
        # ['uniform', '10', '0.5', '1', '1000', '9']
        tog = lambda x: x.replace('stake_growth.csv', '').split('/')[-1].split('_') + [x.split('/')[-2]]
        df['decomp_fname'] = list(df['fname'].apply(tog))        
        
        npart = lambda x: x[1] #lambda x: x.replace('stake_growth.csv', '').split('/')[-1].split('_')[2]
        df['n_participants'] = list(df['decomp_fname'].apply(npart))

        return df

    def process(self, df_lst, distr = ''):

        df = pd.concat(df_lst, ignore_index = True)
        
        # Reconstruct Simulation Parameters from Filename
        df = self.decompose(df)

        grpd = df.groupby(['n_participants'])

        out_dfs = []

        for npart in df['n_participants'].unique():
            print('Evaluation participants sim: ', npart)
            #pdb.set_trace()
            # Process Filename
            sub = grpd.get_group(npart).dropna(axis = 1).drop(columns = {'n_participants', 'fname', 'decomp_fname', 'Unnamed: 0'}) #.describe())#.mean(axis = 0))
            

            # Expected Return per Round is average over rows
            rets_per_round = sub.mean(axis = 1) # Looks decent, is positive around 10%
            
            # Standard Deviation
            std_per_round = sub.std(axis = 1)

            # Sharpe Ratio
            sharpe_ratio = rets_per_round / std_per_round
            print('Sharpe Ratio: ', sharpe_ratio)

            # Retrieve decomposed filename
            decomp_fnames = df.loc[sub.index][['decomp_fname']]

            #out_df = pd.DataFrame({'decomp_fnames': decomp_fnames, 'rets': rets_per_round, 'std' : std_per_round, 'sharperatio': sharpe_ratio})

            out_df = pd.concat([decomp_fnames, rets_per_round, std_per_round, sharpe_ratio], ignore_index = True, axis = 1)#.reset_index()
            out_df = out_df.rename(columns = {0:'decomp_fnames', 1:'rets', 2:'std', 3:'sharperatio'}).reset_index(drop = True)

            # Save Boxplot
            fix, ax = plt.subplots()
            #ax = out_df.boxplot(column = 'sharperatio', return_type = 'axes')
            plt.boxplot(out_df['sharperatio'])
            # Constant y-Axis
            ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel('')
            ax.set_xticks([])
            #x_axis = ax.get_xaxis()
            #x_axis.set_label_text('foo')
            #pdb.set_trace()
            #plt.suptitle('')
            #ax.set_xlabel('')

            # No x-Axis label
            #ax1 = plt.axes()
            #x_axis = ax1.axes.get_xaxis()
            #x_axis.set_label('')
            #x_label = x_axis.get_label()
            #x_label.set_visible(False)
            #pdb.set_trace()
            #plt.show()
            plt.savefig('boxplot_' + str(distr) + '_nparticipants=' + str(npart) + '.png')
            
            out_dfs.append(out_df)
            del out_df

        out_all = pd.concat(out_dfs, ignore_index = True)
        #pdb.set_trace()
        for i in out_all.index:
            distribution, n_stakers, paper_acceptance_probability, inflation_per_iteration, n_rounds, simulation_iteration = out_all.loc[i]['decomp_fnames']
            out_all.loc[i, 'distribution'] = distribution
            out_all.loc[i, 'n_stakers'] = n_stakers
            out_all.loc[i, 'paper_acceptance_probability'] = paper_acceptance_probability
            out_all.loc[i, 'inflation_per_iteration'] = inflation_per_iteration
            out_all.loc[i, 'n_rounds'] = n_rounds
            out_all.loc[i, 'sim_iteration'] = simulation_iteration

        out_all.sort_values('n_rounds').to_csv(str(distr) + "output_for_paper.csv")

        return out_all

    def mk_boxplots(self, df, col, dfname):
        grpd = df.groupby([col])
        # Save Boxplot
        fix, ax = plt.subplot()
        #ax = out_df.boxplot(column = 'sharperatio', return_type = 'axes')
        unique_grps = df[col].unique()
        pdb.set_trace()
        for grp in unique_grps:
            sub = grpd.get_group(grp).dropna(axis = 1) 
            plt.boxplot(sub['sharperatio'])
        # Constant y-Axis
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('')
        ax.set_xticks([])
        #plt.show()
        savename = 'boxplot_' + str(dfname) + '.png'
        print('Saving combined boxplot in: ', savename)
        plt.savefig(savename)


    def sns_boxplots(self, df, col, dfname):
        fix, ax = plt.subplots()
        dd=pd.melt(df,id_vars=[col],value_vars=['sharperatio'], var_name='Sharpe Ratio')
        sns.boxplot(x=col,y='value',data=dd)
        ax.set_ylim(-1.5, 2.5)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('')
        #ax.set_xticks([])
        savename = 'boxplot_' + str(dfname) + '_' + str(col) + '.png'
        print('Saving combined boxplot in: ', savename)
        plt.savefig(savename)

    def summary_stats(self, df, col, dfname, round_n_digits = 4):
        savename = 'summary_stats_' + str(dfname) + '.csv'
        print('Saving combined boxplot in: ', savename)
        df[col].describe().round(round_n_digits).to_csv(savename)

    def combine_boxplots(self):
        """
        Combined Boxplots from saved files (out_all)

        paretooutput_for_paper.csv
        uniformoutput_for_paper.csv
        """

        for saved_df in ['paretooutput_for_paper', 'uniformoutput_for_paper']:

            df = pd.read_csv(saved_df + '.csv')

            # Boxplots
            self.sns_boxplots(df, 'n_rounds', saved_df)
            self.sns_boxplots(df, 'n_stakers', saved_df)

            # Summary Statistics
            self.summary_stats(df, ['rets', 'std', 'sharperatio'], saved_df)
            
            #self.mk_boxplots(df, 'n_rounds', saved_df)
            #self.mk_boxplots(df, 'n_stakers', saved_df)



    def load_and_merge_output(self, tgt_str, out_dir):
        """
        For Pareto and Uniform Distributions only!! 
        tgt_str = 'stake_growth.csv'
        out_dir = 'sim/out/'
        """

        if os.path.exists(out_dir):
            #os.walk(out_dir)
            subdirs = glob(out_dir + '/*/*', recursive = True)

            # Load Stake Growth
            # Distribution, five point summary, how many people got knocked out

            # Stakes over time
            # Estimated time of survival?
            
           
            uniform_dfs = []
            pareto_dfs = []
            for subdir in subdirs:
                print('Loading ', subdir)
            
                if tgt_str in subdir:
                    if 'uniform' in subdir:
                        df = pd.read_csv(subdir)
                        df['fname'] = subdir
                        uniform_dfs.append(df)
                    elif 'pareto' in subdir:
                        df = pd.read_csv(subdir)
                        df['fname'] = subdir
                        pareto_dfs.append(df)
                    else:
                        raise ValueError('No correct distribution in filename')

        self.process(uniform_dfs, 'uniform')
        self.process(pareto_dfs, 'pareto')


    def analyze_output(self, tgt_str, out_dir):
        print(out_dir)
        unif, pareto = self.load_and_merge_output(tgt_str, out_dir)

    def create_output_id(self, distribution, n_stakers, paper_acceptance_probability, inflation_per_iteration, n_rounds):
        # Filename for Output
        output_id = \
        str(distribution) + '_' + \
        str(n_stakers) + '_' + \
        str(paper_acceptance_probability) + '_' + \
        str(inflation_per_iteration) + '_' + \
        str(n_rounds)

        return output_id

    def run(self, 
            distribution,
            n_stakers,
            paper_acceptance_probability, 
            inflation_per_iteration, 
            n_rounds,
            out_dir = 'sim/out/'):
        """
        @n_rounds int, amount of iterations
        """
        
        if not os.path.exists(out_dir):
            print('Couldnt find Output directory, creating ', out_dir)
            os.makedirs(out_dir)



        # Filename for Output
        output_id = \
        str(distribution) + '_' + \
        str(n_stakers) + '_' + \
        str(paper_acceptance_probability) + '_' + \
        str(inflation_per_iteration) + '_' + \
        str(n_rounds)


        # Assign Stakes and initialize Votes to a set of Stakers
        self.stakes, self.votes = self.setup(distribution,
                                            n_stakers,
                                            paper_acceptance_probability, 
                                            inflation_per_iteration, 
                                            n_rounds)

        stake_snapshots = []
        # A paper is being proposed. The vote of a single staker is Bernoulli-distributed.

        iterated_stake = {}
        outdf = pd.DataFrame()

        for i in range(self.n_rounds):            
            new_stakes = self.vote(max_bet_share = 1, balance_losses = True)
            stake_snapshots.append(new_stakes)

        
        out = pd.DataFrame(stake_snapshots)
        
        # Save Results
        out.to_csv(out_dir + output_id + '.csv')
    
        # Do the stats
        # Collect Gini Coefficient over Tim
        #gini_coefficients = []
        #for i in range(out.shape[0]):
        #    gini_coefficients.append(self.gini_coefficient(out[:i].to_numpy()))

        # Stake snapshot in the last round
        tl = out.tail(1)
        print('Final Stake: ', tl)
        tl.to_csv(out_dir + output_id + 'final_stake.csv')

        print('Stakes over Time:', out.describe())
        out.describe().to_csv(out_dir + output_id + 'stakes_over_time.csv')


        # Compare to initial distribution
        initial_stake_df = pd.DataFrame(self.initial_stakes, index = [0])
        print('Initial Stakes: ', initial_stake_df)


        # Stake Growth
        stake_growth = pd.DataFrame((tl.to_numpy() - initial_stake_df.to_numpy())/ initial_stake_df.to_numpy())

        print('Stake Growth: ', stake_growth)
        stake_growth.to_csv(out_dir + output_id + 'stake_growth.csv')

        # Amount of accepted papers
        # Cant divide by zero, force omitted papers to be at least one
        # Save this info in the stake growth
        accepted_paper_share = self.accepted_paper_counter / self.n_rounds
        print('Share of accepted papers: ', accepted_paper_share)
        pd.DataFrame({'accepted_paper_share' : accepted_paper_share}, index = [0]).to_csv(out_dir + output_id + '.csv')




if __name__ == '__main__':

    PRS = PeerReviewSim()
    
    
    n_simulations = 10
    n_stakers = [10, 50, 100, 1000] * 2
    n_rounds = [100, 1000, 10000, 100000] * 2
    half_len = int(len(n_rounds) / 2)
    distributions = (['uniform'] * half_len) + (['pareto'] * half_len) 

    # Output Subdirectory
    simulation_dir = 'prod'

    for sim_number in range(n_simulations):
        for params in zip(n_rounds, n_stakers, distributions):
            
            curr_rounds = params[0]
            curr_stakers = params[1]
            curr_distribution = params[2]

            print('Current Simulation with n_rounds, n_stakers: ', curr_rounds, curr_stakers)
            
            PRS.run(distribution = curr_distribution,
                n_stakers = curr_stakers, 
                paper_acceptance_probability = 0.5, 
                inflation_per_iteration = 1, 
                n_rounds = curr_rounds,
                out_dir = str('sim/out/' + simulation_dir + '/') + str(sim_number) + '/')

            
    PRS.analyze_output(tgt_str ='stake_growth.csv', out_dir = 'sim/out/' + simulation_dir + '/')

    # Boxplots and Descriptive Statistics from saved Output
    PRS.combine_boxplots()

    """
    PRS.run(distribution = 'uniform',
            n_stakers = 10, 
            paper_acceptance_probability = 0.5, 
            inflation_per_iteration = 1, 
            n_rounds = 100)
    """
