import pandas as pd
import matplotlib.pyplot as plt


def plot_trajectories(traj, ax=None):
    " Plot trajectories of individuals "
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('w')
    individuals = traj['ind'].unique()
    for ind in individuals:
        traj_ind = traj.loc[traj['ind'] == ind]
        ax.plot(traj_ind['lon'], traj_ind['lat'],
                linewidth=0.5, marker='o', markersize=3,
                label='individual {}'.format(ind))
        # ax.scatter(traj_ind['lon'], traj_ind['lat'], s=3)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def plot_result(traj, origins, n_origins=10, ax=None, legend=True):
    """ Plot the result of an inference run.

    Parameters:
    -----------
     - traj: pd.DataFrame
        The input trajectories provided to the inference method.

     - origins: pd.DataFrame
        The outbreak origins inferred by the inference method.

     - n_origins: int
        How many of the infered outbreak origins to plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('w')

    plot_trajectories(traj, ax)
    top_origin = origins.head(1)
    top_origins = origins.head(n_origins)
    ax.scatter(top_origin['lon'], top_origin['lat'],
               s=150, marker='o', edgecolor='k', linewidth=2, facecolors='none', zorder=20,
               label='estimated outbreak origin')
    ax.scatter(top_origins['lon'], top_origins['lat'],
               s=100, marker='o', edgecolor='k', linestyle='--', facecolors='none', zorder=10,
               label='outbreak origin\ncandidates')
    if legend:
        ax.legend()


def load_sample_trajectories():
    " Load sample trajectories of 4 individuals generated from a dEPR model "
    # Load simulated trajectories of 1000 individuals
    traj = pd.read_csv('../data/sample_trajectories_N1000.csv')

    # Prepare a smaller sample of trajectories of 4 individuals
    sample_individuals = [0, 1, 3, 4]
    sample_trajectories = traj.loc[traj['ind'].isin(sample_individuals)]
    sample_trajectories = sample_trajectories.loc[sample_trajectories['t'] < 100]
    return sample_trajectories


def convert_trajectories_to_parallel_format(traj_sequential):
    """ Convert data to format better for inference

    This method converts the data to a format that is better
    suited for the inference methods. The inference runs faster
    on the parallelized format.

    The input is also resampled to 15 minute increments, which reduces
    computation time (at cost of temporal accuracy).

    Parameters:
    -----------
        - traj_sequential: pd.DataFrame
            Simulation input trajectories of the format
                ind | t | lon | lat
            where each line is on recorded geo-time stamp.

    Returns:
    --------
        - traj_parallel: pd.DataFrame
            Parallelized dataframe, format:
                        |  user 1   |  user 2   | ...
            timestamp   | lat | lon | lat | lon | ...

    """
    traj_sequential = traj_sequential.assign(
        t=pd.to_datetime(traj_sequential['t'], unit='h'))
    traj_sequential = traj_sequential.set_index('t')

    # Round index to 15 min before parallelizing
    # Otherwise too expensive
    traj_sequential.index = traj_sequential.index.round('15min')
    traj_sequential = traj_sequential.groupby(by=['t', 'ind']).first()
    traj_sequential = traj_sequential.reset_index(level=1)
    # Make parallel
    traj_parallel = traj_sequential.pivot(columns='ind')
    traj_parallel = traj_parallel.T.swaplevel().T
    traj_parallel.columns.rename('individual', level=0, inplace=True)
    traj_parallel.columns.rename('location', level=1, inplace=True)
    # If location of an ind. does not change, use last known location
    traj_parallel = (traj_parallel.fillna(method='ffill')
                                  .fillna(method='bfill'))
    # Reorder columns in correct schema
    traj_parallel = traj_parallel.sort_index(axis=1, level=[0, 1],
                                             ascending=[True, False])
    return traj_parallel
