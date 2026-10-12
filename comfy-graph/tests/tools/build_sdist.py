"""Keep Hatchling's forced parent VCS files outside the standalone sdist."""
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Hatchling force-includes ancestor .gitignore/.hgignore even with an
        # explicit include list. They are repository policy, not package inputs.
        forced = build_data['force_include']
        for source in list(forced):
            if Path(source).name in {'.gitignore', '.hgignore'}:
                del forced[source]
