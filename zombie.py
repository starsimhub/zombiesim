import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd

class Zombie(ss.SIR):
    def __init__(self, pars=None, **kwargs):
        super().__init__() # Don't pass kwargs here, as it raises a value error

        self.default_pars(
            inherit = True, # Inherit from SIR defaults
            dur_inf = ss.constant(v=1000), # Once a zombie, always a zombie! Units are years.

            p_fast = ss.bernoulli(p=0.10), # Fast zombies attack more people!
            dur_fast = ss.constant(v=1000), # Once fast, always fast. Units are years.
            p_symptomatic = ss.bernoulli(p=1.0), # Zombies are symptomatic
            p_death_on_zombie_infection = ss.bernoulli(p=0.25),

            p_death = ss.bernoulli(p=1), # All zombies die instead of recovering
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('fast', default=self.pars.p_fast),
            ss.BoolArr('symptomatic', default=False),
            ss.FloatArr('ti_slow'),
        )

        self.cum_congenital = 0 # Count cumulative congenital cases

        return

    def update_pre(self):
        super().update_pre()

        # Fast to slow transition
        fast_to_slow_uids = (self.infected & self.fast & (self.ti_slow <= self.sim.ti)).uids
        self.fast[fast_to_slow_uids] = False

        return

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses """
        super().set_prognoses(uids, source_uids)

        # Choose which new zombies will be symptomatic
        self.symptomatic[uids] = self.pars.p_symptomatic.rvs(uids)

        # Handle fast-->slow transition
        fast_uids = uids[self.fast[uids]]
        dur_fast = self.pars.dur_fast.rvs(fast_uids)
        self.ti_slow[fast_uids] = np.round(self.sim.ti + dur_fast / self.sim.dt)

        # Handle possible immediate death on zombie infection
        dead_uids = self.pars.p_death_on_zombie_infection.filter(uids)
        self.ti_dead[dead_uids] = self.sim.ti # Immediate death
        return

    def set_congenital(self, target_uids, source_uids=None):
        """ Congenital zombies! """
        self.cum_congenital += len(target_uids)
        self.set_prognoses(target_uids, source_uids)
        return

    def init_results(self):
        super().init_results()
        sim = self.sim
        self.results += [ ss.Result(self.name, 'cum_congenital', sim.npts, dtype=int, scale=True) ]
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        res.cum_congenital[ti] = self.cum_congenital
        return


class DeathZombies(ss.Deaths):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__(death_rate=kwargs.pop('death_rate', None))
        self.default_pars(
            inherit = True,
            p_zombie_on_natural_death = ss.bernoulli(p=0.75), # Susceptible people who die can become zombies
        )
        # Need to remove death_rate due to special handling in the base class
        kwargs.pop('death_rate', None)
        self.update_pars(pars, **kwargs) # Not needed with inherit

        return

    def apply_deaths(self):
        """ Select people to die """

        # Ensure that zombies do not die of natural causes
        not_zombie = self.sim.people.alive.asnew()
        for name, disease in self.sim.diseases.items():
            if 'zombie' in name:
                not_zombie = not_zombie & (~disease.infected)

        death_uids = self.pars.death_rate.filter(not_zombie.uids)
        zombie_uids, death_uids = self.pars.p_zombie_on_natural_death.filter(death_uids, both=True)
        if len(death_uids):
            self.sim.people.request_death(death_uids)
        if len(zombie_uids):
            # If we have fast_zombie and slow_zombie types, choose slow_zombie
            zombie = 'zombie' if 'zombie' in self.sim.diseases else 'slow_zombie' # If multiple types of zombies, assume slow
            self.sim.diseases[zombie].set_prognoses(zombie_uids)
        return len(death_uids)


class KillZombies(ss.Intervention):
    def __init__(self, year, rate, **kwargs):
        self.requires = Zombie
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        super().__init__(**kwargs)

        self.p = ss.bernoulli(p= lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        eligible = ~sim.people.alive.asnew()
        for name, disease in sim.diseases.items():
            if 'zombie' in name:
                eligible = eligible | (disease.infected & disease.symptomatic)
        death_uids = self.p.filter(eligible.uids)

        sim.people.request_death(death_uids)

        return len(death_uids)

class zombie_vaccine(ss.sir_vaccine):
    """
    Create a vaccine product that affects the probability of infection.
    
    The vaccine can be either "leaky", in which everyone who receives the vaccine 
    receives the same amount of protection (specified by the efficacy parameter) 
    each time they are exposed to an infection. The alternative (leaky=False) is
    that the efficacy is the probability that the vaccine "takes", in which case
    that person is 100% protected (and the remaining people are 0% protected).
    
    Args:
        efficacy (float): efficacy of the vaccine (0<=efficacy<=1)
        leaky (bool): see above
    """
    def administer(self, people, uids):        
        if self.pars.leaky:
            people.zombie.rel_sus[uids] *= 1-self.pars.efficacy
        else:
            people.zombie.rel_sus[uids] *= np.random.binomial(1, 1-self.pars.efficacy, len(uids))
        return

    
class ZombieConnector(ss.Connector):
    """ Connect Fast and Slow Zombies """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='Zombie Connector', requires=[Zombie])

        self.default_pars(
            rel_sus = 0
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        """ Specify cross protection between fast and slow zombies """

        ppl = self.sim.people
        fast = self.sim.diseases['fast_zombie']
        slow = self.sim.diseases['slow_zombie']

        fast.rel_sus[ppl.alive] = 1
        fast.rel_sus[slow.infected] = self.pars.rel_sus

        slow.rel_sus[ppl.alive] = 1
        slow.rel_sus[fast.infected] = self.pars.rel_sus

        return

class ZombieAnalyzer(ss.Analyzer):

    def __init__(self, **kwargs):
        self.requires = [Zombie]
        self.data = []
        self.df = None # Created on finalize

        super().__init__(**kwargs)
        return

    def init_results(self):
        super().init_results()
        self.results += ss.Result(self.name, 'n_congenital', self.sim.npts, dtype=int)
        return

    def apply(self, sim):
        super().apply(sim)

        n_congenital = 0
        for name, disease in sim.diseases.items():
            if 'zombie' not in name:
                continue

            just_born = (sim.people.age >=0 ) & (sim.people.age < sim.dt)
            n_congenital += np.count_nonzero(disease.infected[just_born])

            self.data.append([self.sim.year, n_congenital])
        return

    def finalize(self):
        super().finalize()
        self.df = pd.DataFrame(self.data, columns = ['year', 'new_congential'])

        self.df['cum_congenital'] = self.df['new_congential'].cumsum()

        return

    def plot(self):
        import seaborn as sns

        d = pd.melt(self.df, id_vars=['year', 'arm'], var_name='channel', value_name='Value')
        g = sns.relplot(data=d, kind='line', x='year', hue='arm', col='channel', y='Value', palette='Set1', facet_kws={'sharey':False})

        return g.figure