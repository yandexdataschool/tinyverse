from lasagne.layers import get_all_param_values, set_all_param_values

from tinyverse.presets.utils import lazy
from ..experiment import Experiment


class AgentnetExperiment(Experiment):

    def __init__(self,agent,db,sequence_length):
        """
        A preset of tinyverse.experiment.Experiment that works with Lasagne and Agentnet.
        Implements some of the generic methods for you.

        :param db: database (mandatory first param)
        :param sequence_length: how many iterations to make per 1 weight update.
        """
        self.agent = agent
        super(AgentnetExperiment,self).__init__(db,sequence_length)

    @lazy
    def agent_step(self):
        """
        When you first call agent.agent_step(state,*mem), this will compile such a function from agentnet.
        See Experiment.agent_step for function description
        """
        return self.agent.get_react_function()

    def get_all_params(self,**kwargs):
        """a generic method that returns agent weights state."""
        shareds = list(self.agent.agent_states) + self.agent.policy + self.agent.action_layers
        return get_all_param_values(shareds)

    def load_all_params(self,param_values,**kwargs):
        """a generic method that sets agent parameters to given values"""
        shareds = list(self.agent.agent_states) + self.agent.policy + self.agent.action_layers
        return set_all_param_values(shareds, param_values)

