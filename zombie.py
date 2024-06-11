import starsim as ss
import sciris as sc
import numpy as np


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
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('fast', default=self.pars.p_fast),
            ss.BoolArr('symptomatic', default=False),
        )
        return

    def update_pre(self):
        super().update_pre()

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses """
        super().set_prognoses(uids, source_uids)

        # Choose which new zombies will be symptomatic
        self.symptomatic[uids] = self.pars.p_symptomatic.rvs(uids)

        # Handle possible immediate death on zombie infection
        dead_uids = self.pars.p_death_on_zombie_infection.filter(uids)
        self.ti_dead[dead_uids] = self.sim.ti # Immediate death


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
        death_uids = self.pars.death_rate.filter()
        zombie_uids, death_ids = self.pars.p_zombie_on_natural_death.filter(death_uids, both=True)
        self.sim.people.request_death(death_uids)
        self.sim.diseases['zombie'].set_prognoses(zombie_uids)
        return len(death_uids)

class KillZombies(ss.Intervention):
    def __init__(self, year, rate, **kwargs):
        self.requires = Zombie
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        super().__init__(**kwargs)

        self.p = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        zombie = sim.diseases['zombie']
        ppl = sim.people
        eligible_uids = (zombie.symptomatic & ppl.alive).uids
        dead_uids = self.p.filter(eligible_uids)

        zombie.ti_dead[dead_uids] = self.sim.ti # Kill zombies asap!

        return len(dead_uids)

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