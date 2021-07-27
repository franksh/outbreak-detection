import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as multinorm
from outbreak.util import convert_trajectories_to_parallel_format


class Inference():

    def __init__(self,
                 GAUSS_SIGMA=0.1,
                 MAX_NEIGHBORHOOD_SIZE_KM=1,
                 VERBOSE=False,
                 **kwargs
                 ):
        """ The inference class containing the outbreak detection algorithm.

        The inference class can be provided with input data in the form
        of spatial trajectories of individuals. It will then look for
        times when individuals were co-located.

        The outbreak origin is estimated as the time and location
        where most individuals were close to each other.

        Parameters:
        -----------
         - GAUSS_SIGMA: float
            The variance of the gaussian distributions placed at the
            locations of individuals, in kilometers.

            The sum of these distributions yields the objective function.
            Outbreak origins are estimated at the maxima of the
            objective function.

            The value of sigma determines how wide the individual
            distributions are. Low values mean that individuals have to be
            closely co-located. High values mean that individuals can be
            further apart an still considered as part of the same location.
            The value should be adjusted to the spatial scale of interest.
            See accompanying notebooks for an illustration.


         - MAX_NEIGHBORHOOD_SIZE_KM: float
            The maximum distance between two locations so that they
            can still be grouped by DBSCAN, in kilometers. This parameter
            is required by the DBSCAN algorithm.

            The value depends on the input data and desired result.
            Smaller values lead to more distinct locations, and larger
            values to bigger clusters.

            If not explicitly given, this is set to the same value as the
            variance of the Gaussian.

         - VERBOSE: bool
            Whether to print output during the inference.

        """

        # Parameters
        # The sigma of the gaussian distributions
        self.GAUSS_SIGMA = GAUSS_SIGMA
        # The max. distance at which two inferred locations can be grouped
        self.MAX_NEIGHBORHOOD_SIZE_KM = MAX_NEIGHBORHOOD_SIZE_KM

        self.VERBOSE = VERBOSE

    def find_outbreak_origins(self, input_trajectories):
        """ Estimate outbreak origins among the input trajectories.

        This is the main inference function. It is passed a sample of
        trajectories as input. It then determines and returns likely
        outbreak origins among them.

        Parameters:
        -----------
         - input_trajectories: pd.DataFrame
            A sample of individual trajectories. Each row represents
            a timestamped, geolocated observation of an individual.

            Expected format: time | id_individual | lon | lat


        Returns:
        --------
         - origins: pd.DataFrame
            A list of the inferred outbreak origins, sorted by their score,
            rangin from 0 to 1.  A higher score corresponds to a
            higher confidence that it is the outbreak origin.

            The estimed outbreak origin is the first entry, with the highest
            score.

            Format: time | score | lat | lon
        """
        # Run inference
        origin_candidates = self._find_optima(input_trajectories)

        # Cluster the found locations
        origins = self._cluster_optimas(origin_candidates,
                                        self.MAX_NEIGHBORHOOD_SIZE_KM)

        return origins

    # LOCATION INFERENCE PROCEDURE ######################################

    def _find_optima(self, input_trajectories):
        """
        The main inference function,

        Computes the overlap between trajectories at each timestep
        For each timestep, place gaussians on top of all location stamps
        and compute local optima of the aggregated density distribution.

        Parameters:
        -----------
         - input_trajectories: pd.DataFrame
            A sample of individual trajectories. Each row represents
            a timestamped, geolocated observation of an individual.

            Expected format: time | id_individual | lon | lat

        Returns:
        --------
         - origin_candidates: pd.DataFrame
            All the optimas found by the inference. Not that this contains
            many locations multiple times, so a spatial clustering
            of the results should be performed.

            Format: time | score | lat | lon
        """
        traj = convert_trajectories_to_parallel_format(input_trajectories)

        # Extract number of individuals from data
        n_ind = int(traj.shape[1] / 2)

        # Get the variance of the Gaussian in radians
        sigma_rad = self.convert_km_to_rad(self.GAUSS_SIGMA)

        # Maximum global optima val (hypothetical)
        optima_max = (
            multinorm([0, 0], cov=sigma_rad).pdf([0, 0]) * n_ind)

        optimas = []
        if self.VERBOSE:
            print(" - Finding outbreak locations")
        # For each timestep
        for time, vals in self.tqdm_counter(traj.iterrows(),
                                            len(traj)):
            locations = vals.values.reshape(n_ind, 2)
            # Find maxima of all locations, using the Jacobian
            results = np.array([minimize(self.F, jac=self.derF,
                                         method='Newton-CG',
                                         x0=loc, args=(
                                             locations,
                                             sigma_rad,
                                             True),
                                         tol=1)
                                for loc in locations])

            for res in results:
                optima_score = res.fun * -1 / optima_max
                optima = {
                    'time': time,
                    'score': optima_score,
                    'lon': res.x[0],
                    'lat': res.x[1]
                }
                optimas.append(optima)

        origin_candidates = pd.DataFrame(optimas).sort_values(
            by='score', ascending=False)

        # Save
        self.origin_candidates = origin_candidates

        return origin_candidates

    # GAUSSIAN FUNCTIONS ################################
    # Mathematical functions that are used to construct the global
    # optimization function

    def f(self, pos, center, cov):
        """ Return the density of one gaussian at center """
        return multinorm(center, cov=cov).pdf(pos)

    def F(self, pos, locations, cov, inverse=False):
        """ Return the density of gaussians located at [locations] """
        val = sum(self.f(pos, center, cov=cov) for center in locations)
        return val if not inverse else -1 * val

    def derf(self, pos, center, cov):
        """ Derivative of the gaussian, used in the optimization """
        df = - self.f(pos, center, cov) * 1 / cov * (pos - center)
        return df

    def derF(self, pos, locations, cov, inverse=False):
        """ Derivative of gaussians located at [locations] """
        dF = sum(self.derf(pos, center, cov=cov) for center in locations)
        return dF if not inverse else -1 * dF

    # CLUSTERING OF RESULTS ############################

    def _cluster_optimas(self, origin_candidates,
                         max_neighborhood_size_km=0.1):
        """ Cluster the locations in inference result

        origin_candidates contains all locations that are a local/global maximum,
        with their respective score, but many locations are included multiple
        times.

        This function aggregates the locations
        using DBSCAN (because many are likely the same locations) and returns
        the aggregated locations.

        Parameters:
        -----------
         - origin_candidates: pd.DataFrame
            Contains all the local optima found in the inference.
            Format: time | score | lon | lat

         - max_neighborhood_size_km: float
            The maximum distance between two locations so that they
            can still be grouped by DBSCAN, in kilometers. This parameter
            is required by the DBSCAN algorithm.

            The value depends on the input data and desired result.
            Smaller values lead to more distinct locations, and larger
            values to bigger clusters.

        Returns:
        --------
         - origins: pd.DataFrame
            A dataframe containing the most likely outbreak locations,
            sorted by score.
            Format: loc_id | lat | lon | score | time
        """
        # from sklearn import metrics

        if self.VERBOSE:
            print(" - Clustering optimas")

        # Location
        # Smaller -> More distinct locations. Larger -> Fewer, larger clusters

        max_neighborhood_size_rad = self.convert_km_to_rad(
            max_neighborhood_size_km)

        # Get data
        coords = origin_candidates[['lat', 'lon']]

        # Perform clustering and gather labels
        db = DBSCAN(eps=max_neighborhood_size_rad, min_samples=1).fit(coords)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if self.VERBOSE:
            print(" -- Clustering result: {} -> {}".format(
                origin_candidates.shape[0], n_clusters_
            ))

        # Add location labels to stops
        origin_candidates = origin_candidates.copy()
        origin_candidates['loc_id'] = labels

        # Aggregate locations
        origins = (origin_candidates.groupby(by='loc_id').agg(
            {'lat': 'mean', 'lon': 'mean', 'score': 'max'})
            .reset_index().sort_values(by='score', ascending=False))

        # Find average time when at location
        # For the time, get the average time when score is highest
        # at each location
        toptimes = {}
        for loc_id, vals in origins.iterrows():
            # Find instances with top score at location
            dfoptimalscore = origin_candidates[((origin_candidates['loc_id'] == loc_id) &
                                                (origin_candidates['score'] == vals['score']))]
            # Take average
            toptimeint = dfoptimalscore['time'].values.astype(np.int64).mean()

            toptime = pd.to_datetime(int(toptimeint))
            toptimes[loc_id] = toptime
        # Add to result
        origins['time'] = origins['loc_id'].map(toptimes)
        origins = origins.sort_values(by='score', ascending=False)

        self.origins = origins

        return origins

    # UTILITIES ##################################

    def tqdm_counter(self, iterator, total=None):
        # OVERRIDE FOR CLUSTER
        return iterator
        if self.VERBOSE:
            from tqdm import tqdm
            return tqdm(iterable=iterator, total=total)
        else:
            return iterator

    def convert_km_to_rad(self, km):
        """ Convert a quantity given in km to radians """
        kms_per_radian = 6371.0088
        rad = km / kms_per_radian
        return rad
