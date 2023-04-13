import argparse
import sys
import logging

from number_game import games

LOGGER = logging.getLogger(__name__)

GAMES = [
    "ng24",
    "ng77s"
]


class NumberGame:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Number game"
        )
        parser.add_argument('game', help='Game to run')

        # Only parse the first argument (i.e. sys.argv[1:2]), since the other arguments are parsed by the subcommand
        args = parser.parse_args(sys.argv[1:2])
        if args.game not in GAMES:
            parser.print_help()
            raise ValueError(f"Unrecognised game `{args.game}`!")

        # Use dispatch pattern to invoke method with same name
        getattr(self, args.game)()

    @classmethod
    def ng24(cls):
        """ Twenty-four game """
        parser = argparse.ArgumentParser(
            prog='ng24',
            description="Twenty-four game"
        )
        parser.add_argument('integers', type=int, nargs=4,
                            help='Input integers (four numbers)')

        game = games.TwentyFourGame()

        args = parser.parse_args(sys.argv[2:])
        _input = args.integers
        assert len(_input) == 4, "Four numbers should be given!"
        result = game(_input)
        print(result)

    @classmethod
    def ng77s(cls):
        """ Seven sevens game """
        parser = argparse.ArgumentParser(
            prog='ng77s',
            description="Seven sevens game"
        )

        parser.add_argument('start', type=int, default=0, nargs='?',
                            help="Start range")
        parser.add_argument('end', type=int,
                            help="End range")

        game = games.SevenSevensGame()

        args = parser.parse_args(sys.argv[2:])
        assert args.start < args.end, f"start should be before end! {args.start} vs {args.end}"
        targets = list(range(args.start, args.end + 1))
        result = game(targets)
        print(result)


if __name__ == "__main__":
    try:
        NumberGame()
    except Exception as e:
        LOGGER.error(f"FAILED: {e}")
        exit(1)
