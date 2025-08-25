from src.validator import BaseValidator
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--myagent-id', type = str, default = '0')
    parser.add_argument('--myagent-module-path', default = 'Agents/MyAgent')
    parser.add_argument('--opponent-id', type = str, default = '1')
    parser.add_argument('--opponent-module-path', default = 'Agents/BenchMark')
    parser.add_argument('--common-dir', default = 'common')
    parser.add_argument('--youth', default = 0, type = int)

    parser.add_argument('--result-dir', default = './results')
    parser.add_argument('--result-path', default = 'validation_results.json')
    parser.add_argument('--replay', default = 0, type = int)
    parser.add_argument('--num-validation', default = 3, type = int)
    parser.add_argument('--time-out', default = 3.0, type = float)
    parser.add_argument('--memory-limit', default = 7.5, type = float)
    parser.add_argument('--random-test', default = 0, type = int)
    parser.add_argument('--control-init', default = 'random', type = str)
    parser.add_argument('--make-log', default = 1, type = int)
    parser.add_argument('--visualize', default = 0, type = int)
    parser.add_argument('--color', default = 'Red', type = str)
    parser.add_argument('--max-size', default = 0.08, type = float)

    return parser.parse_args()


def main():
    args = parse_args()
    validator = BaseValidator(myagent_id=args.myagent_id,
                              myagent_module_path=args.myagent_module_path,
                              opponent_id=args.opponent_id,
                              opponent_module_path=args.opponent_module_path,
                              common_dir=args.common_dir,
                              youth = args.youth
    )
    validator.validate(result_dir=args.result_dir,
                       result_path=args.result_path,
                       replay=args.replay,
                       num_validation=args.num_validation,
                       time_out=args.time_out,
                       memory_limit=args.memory_limit,
                       random_test=args.random_test,
                       control_init=args.control_init,
                       make_log=args.make_log,
                       visualize=args.visualize,
                       color=args.color,
                       max_size=args.max_size
    )

if __name__ == '__main__':
    main()
