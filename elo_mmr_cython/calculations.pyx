from libc.math cimport tanh, cosh, sqrt, M_PI
from libc.stdint cimport SIZE_MAX

import random
import warnings

from bisect import bisect_left
from itertools import groupby

from numerical import solve_newton, standard_normal_pdf, recip, standard_normal_cdf


cdef public double TANH_MULTIPLIER = M_PI / 1.7320508075688772
cdef public int SECS_PER_DAY = 86_400

# Custom type, which is actually a int but functions as typhint for the enum
ctypedef int OrderingType

cdef enum Ordering:
    LESS = -1
    EQUAL = 0
    GREATER = 1


cdef class PlayerEvent:
    cdef public int contest_index, rating_mu, rating_sig, perf_score
    cdef public int place

    def __cinit__(self, int contest_index, int rating_mu, int rating_sig, int perf_score, int place):
        self.contest_index = contest_index
        self.rating_mu = rating_mu
        self.rating_sig = rating_sig
        self.perf_score = perf_score
        self.place = place

    cdef int get_display_rating(self):
        # TODO: get rid of the magic numbers 3 and 80!
        # 3 is a conservative number of stdevs: use 0 to get mean estimates
        # 80 is Elo-MMR's default sig_lim
        return self.rating_mu - 3 * (self.rating_sig - 80)

    cdef update_rating(self, int rating_mu, int rating_sig):
        self.rating_mu = rating_mu
        self.rating_sig = rating_sig

    def __repr__(self):
        return f"{self.contest_index} {self.rating_mu} {self.rating_sig} {self.perf_score} {self.place}"


cdef class Player:
    cdef public Rating normal_factor
    cdef public list logistic_factors
    cdef public list event_history
    cdef public Rating approx_posterior
    cdef public unsigned long update_time, delta_time

    def __cinit__(self, double mu, double sig, unsigned long long update_time):
        """ Initializes a player with a normal prior and no events

        Args:
            mu {double} -- mean of the normal prior
            sig {double} -- standard deviation of the normal prior
            update_time {unsigned long long} -- time of initialization

        """
        self.normal_factor = Rating(mu, sig)
        self.logistic_factors = []
        self.event_history = []
        self.approx_posterior = Rating(mu, sig)
        self.update_time = update_time
        self.delta_time = 0

    cdef int times_played(self):
        """ Returns the number of events played

        Returns:
            int -- number of events played
        """
        return len(self.event_history)

    cdef int times_played_excl(self):
        """ Returns the number of events played excluding the current one

        Returns:
            int -- number of events played excluding the current one
        """
        return max(0, len(self.event_history) - 1)

    cdef int is_newcomer(self):
        """ Returns whether the player has played no events

        Returns:
            int -- whether the player has played no events
        """
        return self.times_played_excl() == 0

    cdef void update_rating(self, Rating rating, double performance_score):
        """ Updates the player's rating and performance score

        Arguments:
            rating {Rating} -- the player's new rating
            performance_score {double} -- the player's performance score
        """
        last_event = self.event_history[-1]

        # Check that the player's rating is not 0
        # assert last_event.rating_mu == 0
        # assert last_event.rating_sig == 0
        # assert last_event.perf_score == 0

        self.approx_posterior = rating

        # TODO: Why round?
        last_event.rating_mu = round(rating.mu)
        last_event.rating_sig = round(rating.sig)
        last_event.perf_score = round(performance_score)

    cdef void update_rating_with_normal(self, Rating performance):
        cdef double wn = self.normal_factor.sig ** (-2)
        cdef double wp = performance.sig ** (-2)

        self.normal_factor = (wn * self.normal_factor.mu + wp * performance.mu) / (wn + wp)
        self.normal_factor.sig = sqrt(1 / (wn + wp))

        cdef Rating new_rating = self.approximate_posterior(performance.sig) if self.logistic_factors else self.normal_factor

        self.update_rating(new_rating, performance.mu)

    cdef void update_rating_with_logistic(self, Rating performance, unsigned int max_history):
        cdef TanhTerm logistic
        cdef double wn, wl


        if len(self.logistic_factors) >= max_history:
            ## wl can be chosen so as to preserve total weight or rating; we choose the former.
            ## Either way, the deleted element should be small enough not to matter.

            logistic = self.logistic_factors[0]
            self.logistic_factors = self.logistic_factors[1:]

            wn = self.normal_factor.sig ** (-2)
            wl = logistic.get_weight()

            self.normal_factor.mu = (wn * self.normal_factor.mu + wl * logistic.mu) / (wn + wl)
            self.normal_factor.sig = sqrt(1 / (wn + wl))

        self.logistic_factors.append(performance.to_tanh_term())

        cdef Rating new_rating = self.approximate_posterior(performance.sig)

        self.update_rating(new_rating, performance.mu)


    cdef Rating approximate_posterior(self, double perf_sig):
        cdef double normal_weight = self.normal_factor.sig ** (-2)
        cdef double mu = robust_average(
            self.logistic_factors,
            -self.normal_factor.mu * normal_weight,
            normal_weight
        )

        cdef double sig = sqrt(1 / (self.approx_posterior.sig ** (-2) + perf_sig ** (-2)))

        return Rating(mu, sig)

    cdef void add_noise_and_collapse(self, double sig_noise):
        """ Method #1: the Gaussian/Brownian approximation, in which rating is a Markov state
        Equivalent to method #5 with transfer_speed == f64::INFINITY"""
        self.approx_posterior = self.approx_posterior.with_noise(sig_noise)
        self.normal_factor = self.approx_posterior
        self.logistic_factors = []

    cdef void add_noise_best(self, double sig_noise, double transfer_speed):
        """
        #5: a general method with the nicest properties, parametrized by transfer_speed >= 0
        Reduces to method #1 when transfer_speed == f64::INFINITY
        Reduces to method #2 when transfer_speed == 0
        """

        cdef Rating new_posterior = self.approx_posterior.with_noise(sig_noise)
        cdef double decay = (self.approx_posterior.sig / new_posterior.sig) ** 2

        cdef double transfer = decay ** 2

        self.approx_posterior = new_posterior

        cdef double wt_norm_old = self.normal_factor.sig ** (-2)
        cdef double wt_from_norm_old = wt_norm_old * transfer
        cdef double wt_from_transfers = (1 - transfer) * (wt_norm_old + sum(l.get_weight() for l in self.logistic_factors))

        cdef double wt_total = wt_from_norm_old + wt_from_transfers

        self.normal_factor.mu = (wt_from_norm_old * self.normal_factor.mu + wt_from_transfers * self.approx_posterior.mu) / wt_total


        self.normal_factor.sig = sqrt(recip(decay * wt_total))
        for r in self.logistic_factors:
            r.w_out *= transfer * decay

    def __repr__(self) -> str:
        return f"{self.approx_posterior.mu} {self.approx_posterior.sig}"

            

cdef class Term:
    cpdef tuple eval(self, double x, OrderingType order, int split_ties):
        raise NotImplementedError("eval not implemented at base class")    
        
    cpdef tuple evals(self, double x, list ranks, int my_rank, int split_ties):
        
        if len(ranks) == 1:
            order = create_ordering(ranks[0], my_rank)
            return self.eval(x, order, split_ties)

        cdef int start, end, equal, greater
        cdef double v, p
        cdef double value, deriv

        value = 0
        deriv = 0

        start, end = equal_range(ranks, my_rank)
        equal = end - start
        greater = len(ranks) - end
        if start > 0:
            v, p = self.eval(x, Ordering.LESS, split_ties)
            value += v * start
            deriv += p * start
        if equal > 0:
            v, p = self.eval(x, Ordering.EQUAL, split_ties)
            value += v * equal
            deriv += p * equal
        if greater > 0:
            v, p = self.eval(x, Ordering.GREATER, split_ties)
            value += v * greater
            deriv += p * greater

        return value, deriv


cdef class Rating(Term):
    cdef public double mu, sig

    def __cinit__(self, double mu, double sig):
        self.mu = mu
        self.sig = sig

    cdef Rating with_noise(self, double sig_noise):
        return Rating(self.mu, (self.sig ** 2 + sig_noise ** 2) ** 0.5)

    cdef Rating towards_noise(self, double decay, Rating limit):
        cdef double mu_diff = self.mu - limit.mu
        cdef double sig_sq_diff = self.sig ** 2 - limit.sig ** 2
        return Rating(limit.mu + mu_diff * decay, (limit.sig ** 2 + sig_sq_diff * decay ** 2) ** 0.5)

    cpdef tuple eval(self, double x, OrderingType order, int split_ties):
        cdef double z, pdf, pdf_prime, cdf, cdf_m1, val, val_m1, pdf_pp

        z = (x - self.mu) / self.sig
        pdf = standard_normal_pdf(z) / self.sig
        pdf_prime = -z * pdf / self.sig

        if order == Ordering.LESS:
            cdf_m1 = -standard_normal_cdf(-z)
            val = pdf / cdf_m1
            return val, pdf_prime / cdf_m1 - val ** 2
        elif order == Ordering.GREATER:
            # Evaluate first when not splitting ties
            if not split_ties:
                pdf_pp = -(pdf / self.sig + z * pdf_prime) / self.sig
                val = pdf_prime / pdf
                return val, pdf_pp / pdf - val ** 2

            # When splitting ties
            cdf = standard_normal_cdf(z)
            cdf_m1 = -standard_normal_cdf(-z)
            val = pdf / cdf
            val_m1 = pdf / cdf_m1
            return (
                0.5 * (val + val_m1),
                0.5 * (pdf_prime * (1 * cdf + 1 * cdf_m1) - val ** 2 - val_m1 ** 2)
            )
        else:
            cdf = standard_normal_cdf(z)
            val = pdf / cdf
            return pdf, pdf_prime / cdf - val ** 2

    cpdef TanhTerm to_tanh_term(self):
        cdef double w = TANH_MULTIPLIER / self.sig
        return TanhTerm(self.mu, w * 0.5, w)


cpdef list get_participant_ratings(dict players, tuple contest_standings,
                                    unsigned long long min_history):
    """ Returns the ratings of the participants in a contest

    Arguments:
        players {dict} -- dictionary of players
        contest_standings {tuple} -- tuple of contest standings
        min_history {unsigned long long} -- minimum number of events played

    Returns:
        tuple -- tuple of ratings
    """

    cdef list standings = []
    cdef Player player

    # Collect the ratings of the participants
    # Add to standings if the player has played enough events
    # Added as a tuple as (approx_posterior, low, high)
    for reference, low, high in contest_standings:
        player = players[reference]

        if player.times_played() >= min_history:
            standings.append((player.approx_posterior, low, high))

    ## Normalizing the ranks is very annoying, I probably should've just represented
    ## standings as an Vec of Vec of players
    cdef unsigned int last_k_low, last_v_low, last_k_high, last_v_high
    for i, (_, low, high) in enumerate(standings):
        if low != last_k_low:
            last_k_low = low
            last_v_low = i
        if high != last_k_high:
            last_k_high = high
            last_v_high = i

        # Change the low and high ranks to the index of the player
        standings[i][1] = last_v_low
        standings[i][2] = last_v_high

    return standings


cdef class TanhTerm(Term):
    cdef public double mu, w_arg, w_out

    def __cinit__(self, double mu, double w_arg, double w_out):
        self.mu = mu
        self.w_arg = w_arg
        self.w_out = w_out

    @staticmethod
    cdef TanhTerm from_rating(Rating rating):
        cdef double w = TANH_MULTIPLIER / rating.sig
        return TanhTerm(rating.mu, w * 0.5, w)

    cpdef double get_weight(self):
        return self.w_arg * self.w_out * 2. / (TANH_MULTIPLIER ** 2)

    cdef tuple base_values(self, double x):
        """ Returns the base values of the tanh term. This is the value of the
        tanh term and its derivative with respect to the input value.

        Arguments:
            x {double} -- the input value

        Returns:
            tuple -- the base value and its derivative
        """
        cdef double z = (x - self.mu) * self.w_arg
        cdef double val = tanh(-z) * self.w_out
        cdef double val_prime = cosh(-z) ** (-2) * self.w_arg * self.w_out
        return val, val_prime

    cpdef tuple eval(self, double x, OrderingType order, int split_ties):
        """ Returns the value of the tanh term and its derivative with respect
        to the input value.

        Arguments:
            x {double} -- the input value
            order {Ordering} -- the ordering of the input value
            split_ties {int} -- whether to split ties

        Returns:
            tuple -- the value and its derivative
        """
        cdef double val, val_prime

        val, val_prime = self.base_values(x)

        if order == Ordering.LESS:
                return val - self.w_out, val_prime
        elif order == Ordering.GREATER:
            if not split_ties:
                return 2 * val, 2 * val_prime
            
            return val, val_prime
        else:
            return val + self.w_out, val_prime

    cpdef tuple evals(self, double x, list ranks, int my_rank, int split_ties):
        cdef OrderingType order

        if len(ranks) == 1:
            order = create_ordering(ranks[0], my_rank)
            return self.eval(x, order, split_ties)
        
        cdef double val, val_prime
        cdef int start, end, total, win_minus_loss, equal

        val, val_prime = self.base_values(x)

        start, end = equal_range(ranks, my_rank)

        total = len(ranks)
        win_minus_loss = total - (start + end)

        if not split_ties:
            equal = end - start
            total += equal

        value = val * total + self.w_out * win_minus_loss
        deriv = val_prime * total

        return (value, deriv)


cdef class RatingSystem:

     def __cinit__(self):
         pass


cdef class ContestRatingParams:
    """ Parameters for calculating contest rating.

    Attributes:
        weight {int} -- weight of the contest in the rating calculation. Default: 1
        perf_ceiling {float} -- maximum performance in the contest. Default: infinity

    """

    cdef public int weight
    cdef public float perf_ceiling

    def __cinit__(self, int weight=1, float perf_ceiling=float('inf')):
        self.weight = weight
        self.perf_ceiling = perf_ceiling


cdef class Contest:

    cdef public unicode name
    cdef public str url
    cdef public ContestRatingParams rating_params
    cdef public unsigned int time_seconds
    cdef public list standings

    def __cinit__(self, int index):
        self.name = f"Round #{index}"
        self.url = ""
        self.rating_params = ContestRatingParams()
        self.time_seconds = index * 86_400
        self.standings = []

    cdef int find_contestant(self, str handle):
        """ Finds contestant in the standings if they participated in the contest.
        Returns -1 if the contestant did not participate.

        Arguments:
            handle {str} -- handle of the contestant

        Returns:
            int -- index of the contestant in the standings
        """
        for i, (h, _, _) in enumerate(self.standings):
            if h == handle:
                return i
        
        return -1

    cdef tuple remove_contestant(self, str handle):
        """ Removes contestant from the standings if they participated in the contest.
        Returns None if the contestant did not participate.

        Arguments:
            handle {str} -- handle of the contestant

        Returns:
            tuple -- (index of the contestant in the standings, contestant's rating, contestant's rank)
        """
        cdef int position = self.find_contestant(handle)
        
        # If the contestant did not participate, return empty tuple
        if position == -1:
            return tuple()

        # Remove contestant from the standings
        contestant = self.standings.pop(position)

        cdef int i
        # Update ranks in the standings
        for i, (_, low, high) in enumerate(self.standings):
            if high >= position:
                self.standings[i][2] -= 1
                if low > position:
                    self.standings[i][1] -= 1
        
        return contestant

    cdef fix_low_high(self):
        ## Assuming `self.standings` is a subset of a valid standings list,
        ## corrects the `lo` and `hi` values to make the new list valid
        

        # Sort standings by low value
        self.standings.sort(key=lambda x: x[1])

        cdef int number_of_contestants = len(self.standings)
        cdef int lo = 0
        cdef int i, hi

        while lo < number_of_contestants:
            hi = lo + 1
            while (hi + 1) < number_of_contestants and self.standings[lo][1] == self.standings[hi + 1][1]:
                hi += 1
            
            for i in range(lo, hi + 1):
                self.standings[i][1] = lo
                self.standings[i][2] = hi

            lo = hi + 1

    cdef Contest clone_with_standings(self, list standings):
        """ Creates a clone of the contest with the given standings.

        Arguments:
            standings {list} -- list of standings

        Returns:
            Contest -- clone of the contest with the given standings
        """
        cdef Contest contest = Contest(0)
        contest.name = self.name
        contest.url = self.url
        contest.rating_params = self.rating_params
        contest.time_seconds = self.time_seconds
        contest.standings = standings
        contest.fix_low_high()
        return contest
    
    cdef tuple random_split(self, int n):
        """ Splits the contest into n random contests with the same standings."""
        cdef int number_of_standings, chunks

        random.shuffle(self.standings)
        number_of_standings = len(self.standings)
        chunks = number_of_standings // n
        if number_of_standings % n:
            chunks += 1
        return tuple(self.clone_with_standings(self.standings[i*n : (i+1)*n]) for i in range(chunks))
        
    cpdef void push_contestant(self, str handle):
        """ Add a contestant with the given handle in last place.

        Arguments:
            handle {str} -- handle of the contestant
        """
        cdef int number_of_contestants = len(self.standings)
        self.standings.append((handle, number_of_contestants, number_of_contestants))


cdef class ContestSummary:

    cdef public unicode name
    cdef public str url
    cdef public ContestRatingParams rating_params
    cdef public unsigned int time_seconds
    cdef public list standings

    def __cinit__(self, Contest contest):
        self.name = contest.name
        self.url = contest.url
        self.rating_params = contest.rating_params
        self.time_seconds = contest.time_seconds
        self.standings = contest.standings


cdef class EloMMRVariant:
    pass


cdef class Gaussian(EloMMRVariant):
    cdef public float weight

    def __cinit__(self):
        self.weight = float('inf')


cdef class Logistic(EloMMRVariant):

    cdef public float weight

    def __cinit__(self, double weight):
        self.weight = weight


cdef class EloMMR:
    """EloMMR class for calculating Elo ratings and MMRs

    Attributes:
        weight limit {double} --  the weight of each new contest
        noob delay {double} -- weight multipliers (less than one) to apply on first few contests
        siglimit {double} -- each contest participation adds an amount of drift such that, in the absence of
        much time passing, the limiting skill uncertainty's square approaches this value
        drift_per_day {double} -- additional variance per day, from a drift that's continuous in time
        split_ties {int} -- whether to count ties as half a win plus half a loss (1) or as a win and a loss (0)
        subsample_size {unsigned int} --  maximum number of opponents and recent events to use, as a compute-saving approximation
        subsample_bucket {double} --  width of mu and sigma to group subsamples by
        variant {EloMMRVariant} -- the variant to use
    
    """
    
    cdef public double weight_limit
    cdef public list noob_delay
    cdef public double sig_limit
    cdef public double drift_per_day
    cdef public bint split_ties
    cdef public unsigned int subsample_size
    cdef public double subsample_bucket
    cdef public EloMMRVariant variant
    cdef object sorting

    def __cinit__(self, double weight_limit, double sig_limit, bint split_ties, bint fast, EloMMRVariant variant):
        """Initialize EloMMR class"""
        assert weight_limit > 0, "Weight limit should be positive"
        assert sig_limit > 0, "Sig limit should be positive"

        cdef int subsample_size = 100 if fast else <int>float('inf')
        cdef double subsample_bucket = <double>(2.0 if fast else 1e-5)

        # Assign values
        self.weight_limit = weight_limit
        self.noob_delay = []
        self.sig_limit = sig_limit
        self.drift_per_day = <double>0.0
        self.split_ties = split_ties
        self.subsample_size = subsample_size
        self.subsample_bucket = subsample_bucket
        self.variant = variant

        self.sorting = sort_on_mu_sig_rank(self.subsample_bucket)

    cdef double compute_weight(self, double contest_weight, unsigned int n):
        contest_weight *= self.weight_limit

        # Check if n is within the index bounds of self.noob_delay
        if self.noob_delay and len(self.noob_delay) - 1 <= n:
            contest_weight *= self.noob_delay[n]
        
        return contest_weight

    cdef double compute_sig_perf(self, double weight):
        cdef double discrete_perf = (1 + 1 / weight) * self.sig_limit ** 2
        cdef double continuous_perf = self.drift_per_day / weight

        return sqrt(discrete_perf + continuous_perf)

    cdef double compute_sig_drift(self, double weight, double delta_seconds):
        cdef double discrete_drift = weight * self.sig_limit ** 2
        cdef double continuous_drift = self.drift_per_day * delta_seconds / SECS_PER_DAY

        return sqrt(discrete_drift + continuous_drift)

    
    cdef object subsample(self, list terms, double rating, unsigned int num_samples, double subsample_bucket):
        terms = sorted(terms, key=lambda x: x[0].mu)





        beg = binary_search_by(terms, rating, subsample_bucket)
        end = beg + 1

        expand = (num_samples - max(0, end - beg) + 1) // 2
        beg = max(0, beg - expand)
        end = min(len(terms), end + expand)

        expand = num_samples - max(0, end - beg)
        beg = max(0, beg - expand)
        end = min(len(terms), end + expand)

        return range(beg, end)

    cpdef list round_update(self, ContestRatingParams rating_params, list standings):
        """Update ratings for a single round.

        Update ratings due to waiting period between contests,
        then use it to create Gaussian terms for the Q-function.
        The rank must also be stored in order to determine if it's a win, loss, or tie
        term. filter_map can exclude the least useful terms from subsampling.

        Arguments:
            rating_params {ContestRatingParams} -- rating parameters
            standings {tuple} -- standings of the round
        """
        cdef double weight, sig_perf, sig_drift
        cdef list base_terms, normal_terms, tanh_terms, ranks
        cdef Rating last_term
        cdef unsigned int number_of_ranks, my_rank, idx_len_upper_bound
        cdef int idx_len_max
        cdef double player_mu, mu_perf
        cdef object idx_subsample, f
        cdef Player player
        cdef tuple bounds

        base_terms = []

        for (player, low, _) in standings:
            weight = self.compute_weight(rating_params.weight, player.times_played())
            sig_perf = self.compute_sig_perf(weight)
            sig_drift = self.compute_sig_drift(weight, player.delta_time)

            if isinstance(self.variant, Logistic) and self.variant.weight < float('inf'):
                player.add_noise_best(sig_drift, self.variant.weight)
            else:
                player.add_noise_and_collapse(sig_drift)
            
            base_terms.append((player.approximate_posterior(sig_perf), low))
            

            # Bucket by 'mu', then by 'sigma', then by 'rank'
            base_terms = sorted(base_terms, key=self.sorting)
            # base_terms = sorted(base_terms, key=lambda x: (x[0].mu, x[0].sig, x[1])                 

            normal_terms = []

            ## Sort terms by rating to allow for subsampling within a range or ratings.
            for (term, low) in base_terms:
                if len(normal_terms) == 0:
                    normal_terms.append((term, [low]))
                    continue

                last_term, ranks = normal_terms[-1]
                if same_bucket(last_term.mu, term.mu, self.subsample_bucket) and same_bucket(last_term.sig, term.sig, self.subsample_bucket):
                    number_of_ranks = len(ranks)
                    last_term.mu = (number_of_ranks * last_term.mu + term.mu) / (number_of_ranks + 1)
                    last_term.sig = (number_of_ranks * last_term.sig + term.sig) / (number_of_ranks + 1)
                    ranks.append(low)
                    continue

                normal_terms.append((term, [low]))

            # Create the equivalent logistic terms.
            tanh_terms = []
            for (rating, ranks) in normal_terms:
                tanh_terms.append((rating.to_tanh_term(), ranks))

            idx_len_max = 9999

            ## The computational bottleneck: update ratings based on contest performance
            for (player, my_rank, _) in standings:
                player_mu = player.approx_posterior.mu
                idx_subsample = self.subsample(normal_terms, player_mu, 
                                               self.subsample_size, self.subsample_bucket)
                idx_len_upper_bound = len(idx_subsample) if idx_subsample is not None else SIZE_MAX

                if idx_len_max < idx_len_upper_bound:
                    idx_len_max = idx_len_upper_bound
                    warnings.warn("Subsampling %ld opponents might be slow; consider decreasing subsample_size." % idx_len_upper_bound)

                bounds = (-6000, 9000)
                weight = self.compute_weight(rating_params.weight, player.times_played_excl())
                sig_perf = self.compute_sig_perf(weight)

                if isinstance(self.variant, Gaussian):

                    f = get_formula([normal_terms[i] for i in idx_subsample], my_rank, self.split_ties)

                    mu_perf = solve_newton(bounds, f)
                    player.update_rating_with_normal(
                        Rating(mu_perf, sig_perf)
                    )
                elif isinstance(self.variant, Logistic):


                    f = get_formula([tanh_terms[i] for i in idx_subsample], my_rank, self.split_ties)

                    mu_perf = min(solve_newton(bounds, f), rating_params.perf_ceiling)
                    player.update_rating_with_logistic(
                        Rating(mu_perf, sig_perf), 
                        self.subsample_size
                    )

        return standings

cdef object get_formula(list idx_subsample_copy, int my_rank, bint split_ties):
    """Return callable that can be passed as formula into the solve_newton

    Args:
        idx_subsample_copy {list} -- 

    """

    def f(double x):
        cdef double[2] temp
        cdef double[2] res = [0., 0.]
        for rating, ranks in idx_subsample_copy:
            temp = rating.evals(x, ranks, my_rank, split_ties)
            res[0] += temp[0]
            res[1] += temp[1]
        return tuple(res)

    return f

cpdef EloMMR construct_elo_mmr_default_fast():
    return EloMMR(
        weight_limit=0.2,
        sig_limit=80,
        split_ties=0,
        fast=1,
        variant=Logistic(1.0)
    )

cpdef EloMMR construct_elo_mmr_default_gaussian():
    return EloMMR(
        weight_limit=0.2,
        sig_limit=80,
        split_ties=0,
        fast=0,
        variant=Gaussian()
    )

cpdef EloMMR construct_elo_mmr_gaussian_fast():
    return EloMMR(
        weight_limit=0.2,
        sig_limit=80,
        split_ties=0,
        fast=1,
        variant=Gaussian()
    )


cdef tuple equal_range(list ranks, int my_rank):
    """ Equivalent to ranks.equal_range(my_rank). Get the start and end 
    indexes of the players with the same rank as the player.

    Arguments:
        ranks {list} -- list of ranks
        my_rank {int} -- rank of the player

    Returns:
        tuple -- range of indices of the players with the same rank
    """
    cdef int start, end
    cdef list indexes_equal_rank

    indexes_equal_rank = [i for i, x in enumerate(ranks) if x == my_rank]
    start = min(indexes_equal_rank) if indexes_equal_rank else 0
    end = max(indexes_equal_rank) if indexes_equal_rank else 0

    return start, end


cdef int bucket(double a, double width):
    return round(a / width)

cdef int same_bucket(double a, double b, double width):
    return bucket(a, width) == bucket(b, width)

cdef OrderingType cmp_by_bucket(double a, double b, double width):
    return create_ordering(bucket(a, width), bucket(b, width))

cpdef object wrapper():
    return sort_on_mu_sig_rank(9)

cdef object sort_on_mu_sig_rank(double subsample_bucket):
    """Convert a cmp= function into a key= function"""
    cdef object func = cmp_by_bucket
    
    class K(object):
        __slots__ = ['obj']
        def __init__(self, tuple obj):
            self.obj = obj
        def __lt__(self, other):
            if func(self.obj[0].mu, other.obj[0].mu, subsample_bucket) != 0:
                return func(self.obj[0].mu, other.obj[0].mu, subsample_bucket) < 0
            elif func(self.obj[0].sig, other.obj[0].sig, subsample_bucket) != 0:
                return func(self.obj[0].sig, other.obj[0].sig, subsample_bucket) < 0
            else:
                return self.obj[1] < other.obj[1]
        def __gt__(self, other):
            if func(self.obj[0].mu, other.obj[0].mu, subsample_bucket) != 0:
                return func(self.obj[0].mu, other.obj[0].mu, subsample_bucket) > 0
            elif func(self.obj[0].sig, other.obj[0].sig, subsample_bucket) != 0:
                return func(self.obj[0].sig, other.obj[0].sig, subsample_bucket) > 0
            else:
                return self.obj[1] > other.obj[1]
        def __eq__(self, other):
            return func(self.obj[0].mu, other.obj[0].mu, subsample_bucket) == 0 and func(self.obj[0].sig, other.obj[0].sig, subsample_bucket) == 0 and self.obj[1] == other.obj[1]


        def __le__(self, other):
            return self < other or self == other
        def __ge__(self, other):
            return self > other or self == other
        __hash__ = None
        
    return K

cdef OrderingType create_ordering(int a, int b):
    if a < b:
        return Ordering.LESS
    elif a == b:
        return Ordering.EQUAL
    else:
        return Ordering.GREATER


cdef double robust_average(list all_ratings, double offset, double slope):
    """ Returns the robust average of a list of ratings

    Arguments:
        all_ratings {list} -- list of ratings
        offset {double} -- offset of the robust average
        decay {double} -- decay of the robust average

    Returns:
        double -- robust average
    """
    cdef tuple bounds = (-6000.0, 9000.0)

    def f(double x):
        sum_tanh = offset + slope * x
        sum_term = slope
        for i in range(len(all_ratings)):
            tanh_w = tanh((x - all_ratings[i].mu) * all_ratings[i].w_arg)
            sum_tanh += tanh_w * all_ratings[i].w_out
            sum_term += (1. - tanh_w * tanh_w) * all_ratings[i].w_arg * all_ratings[i].w_out
        return sum_tanh, sum_term

    return solve_newton(bounds, f)



cdef int outcome_free(standings):
    """ Returns whether the contest is outcome free

    Arguments:
        standings {list} -- list of standings

    Returns:
        int -- whether the contest is outcome free
    """
    return len(standings) == 0 or standings[0][2] + 1 >= len(standings)


cpdef void simulate_contest(dict players, Contest contest, EloMMR system, 
                            double mu_newbie, double sig_newbie, 
                            unsigned long long contest_index):
    """ Simulates a contest

    Arguments:
        players {dict} -- dictionary of players
        contest {Contest} -- contest to simulate
        system {RatingSystem} -- rating system to use
        mu_newbie {double} -- mean of the normal prior for new players
        sig_newbie {double} -- standard deviation of the normal prior for new players
        contest_index {unsigned long long} -- index of the contest

    """

    if outcome_free(contest.standings):
        warnings.warn(f"Ignoring contest {contest_index} because all players tied")
        return


    # If a player is competing for the first time, initialize with a default rating
    for (reference, _, _ ) in contest.standings:
        if reference not in players:
            players[reference] = Player(mu_newbie, sig_newbie, contest.time_seconds)

    # Verify that handles are distinct and store guards so that the cells
    # can be released later. This setup enables safe parallel processing.
    # TODO: to skip or not to skip?
    # cdef list guards = []
    # for (reference, _, _) in contest.standings:
    #     if reference in players:
    #         guards.append(players[reference])

    # Update player metadata and get references to all requested players
    cdef list standings = []

    for (reference, low, high) in contest.standings:
        player = players[reference]
        player.event_history.append(
            PlayerEvent(contest_index, 
                        0,  ##  will be filled by system.round_update()
                        0,  ##  will be filled by system.round_update()
                        0,  ##  will be filled by system.round_update()
                        low
            )
        )
        player.delta_time = contest.time_seconds - player.update_time;
        player.update_time = contest.time_seconds;

        standings.append((player, low, high))

    system.round_update(contest.rating_params, standings)


cdef int binary_search_by(list terms, double rating, double subsample_bucket):
    cdef int lo, mid, hi
    cdef double mu

    lo = 0
    hi = len(terms)

    while lo < hi:
        mid = (lo+hi)//2
        mu = terms[mid][0].mu
        if cmp_by_bucket(mu, rating, subsample_bucket) > 0:
            hi = mid
        else:
            lo = mid + 1

    return lo
