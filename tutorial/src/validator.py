import os
import json
import tracemalloc

from .utils import make_match_info, total_size
from .fight import Fight


class Validator():
    myagent_id: str
    myagent_module_path: str
    opponent_id: str
    opponent_module_path: str
    commom_dir: str
    youth: bool
    agent_info: dict[str, str | dict]
    opponent_info: dict[str, str | dict]

    def __init__(self,
                 myagent_id: str,
                 myagent_module_path: str,
                 opponent_id: str,
                 opponent_module_path: str,
                 common_dir: str,
                 youth: bool
        ) -> None:

        self.myagent_id = myagent_id
        self.myagent_module_path = myagent_module_path
        self.opponent_id = opponent_id
        self.opponent_module_path = opponent_module_path

        self.agent_info = self._get_agent_info(myagent_id, myagent_module_path)
        self.opponent_info = self._get_agent_info(opponent_id, opponent_module_path)

        self.common_dir = common_dir
        self.youth = youth


    def validate(self) -> None:
        raise NotImplementedError


    def _get_agent_info(self, agent_id:str, module_path:str)->dict:
        module_name = '.'.join(module_path.split(os.path.sep))
        with open(os.path.join(module_path, 'args.json')) as f:
            args = json.load(f)
        agent_info = {
            'userModelID': agent_id,
            'userModuleID': module_name,
            'args': args
        }
        return agent_info


class BaseValidator(Validator):
    def validate(self,
                 result_dir: str,
                 result_path: str,
                 replay: bool,
                 num_validation: int,
                 time_out: float,
                 memory_limit: float,
                 random_test: bool,
                 control_init: str,
                 make_log: bool,
                 visualize: bool,
                 color: str,
                 max_size: float
        ) -> None:
        print('\nValidation:')
        print('random test: {}\n'.format(random_test))

        # fight against the benchmark
        results_dir = os.path.join(result_dir, self.myagent_id)
        os.makedirs(results_dir, exist_ok=True)
        results = self._fight(results_dir, replay, time_out, num_validation, memory_limit, random_test, control_init, make_log, visualize, color)

        # save the results
        for i, log in results['details'].items():
            if not len(log['logs']):
                del log['logs']
                continue
            s = total_size(log['logs'][color])
            if s <= max_size*(1024**3):
                print('Log size for fight {}: {} [GB]'.format(i, s/(1024**3)))
                with open(os.path.join(results_dir, 'log_{}.json'.format(i)), 'w', encoding='utf-8') as f:
                    json.dump(log['logs'][color], f, indent=1)
            else:
                print('Log size for fight {} is too large: {} [GB](>={}[GB])'.format(i+1, s/(1024**3), max_size))

            del log['logs']

        results_path = os.path.join(results_dir, result_path)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=1)
        print('\nSaved the result to {}\n'.format(results_path))


    def _fight(self, results_dir:str, replay:bool, time_out:float, num_validation:int, memory_limit:float, random_test:bool, control_init:str, make_log: bool, visualize: bool, color:str)->dict:
        memory_limit = memory_limit * (1024**3)
        if color=='Red':
            print('You are Red')
            red = self.agent_info
            blue = self.opponent_info
            color_opponent = 'Blue'
        elif color == 'Blue':
            print('You are Blue')
            red = self.opponent_info
            blue = self.agent_info
            color_opponent = 'Red'
        else:
            print('You are Red')
            color = 'Red'
            red = self.agent_info
            blue = self.opponent_info
            color_opponent = 'Blue'

        match_info = make_match_info(red = red,
                                     blue = blue,
                                     replay = replay,
                                     movie_dir = results_dir,
                                     time_out = time_out,
                                     common_dir = self.common_dir,
                                     control_init = control_init,
                                     youth = self.youth,
                                     make_log = make_log,
                                     visualize = visualize)
        details = {}
        num_wins = 0
        num_losses = 0
        num_draw = 0
        fighting = Fight(match_info)
        fighting.make_matchInfo()
        fighting.set_match()

        status = 'success'
        message = 'ok'
        tracemalloc.start()
        for i in range(num_validation):
            print('\n---fight {}---'.format(i+1))
            winner, detail = fighting.run(random_test)
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics('lineno')
            total_memory = 0
            for stat in stats:
                total_memory += stat.size
            print('match {}: {} [Bytes] so far'.format(i, total_memory))

            details[i] = detail

            if total_memory >= memory_limit:
                m = 'total memory: {} [Bytes](>={})'.format(total_memory, memory_limit)
                print(m)
                status = 'error'
                message = str(MemoryLimitExceed('Total memory exceeded the limit. ({})'.format(m)))
                break
            tracemalloc.clear_traces()

            if 'error' in detail and detail['error']:
                status = 'error'
                if color=='Red':
                    message = detail['red_error_message']
                else:
                    message = detail['blue_error_message']
                break


            if winner == color:
                num_wins += 1
            elif winner == color_opponent:
                num_losses += 1
            elif winner == '':
                num_draw += 1
            print('match {}: Red {} vs Blue {}, winner {}'.format(i, self.agent_info['userModelID'], self.opponent_info['userModelID'], winner))
        print('\nnum fights: {}'.format(num_validation))

        results = {
            'own':color,
            'details': details,
            'win':num_wins,
            'loss':num_losses,
            'draw':num_draw,
            'status': status,
            'message': message,
        }

        return results


class MemoryLimitExceed(Exception):
    pass


class ActionError(Exception):
    pass