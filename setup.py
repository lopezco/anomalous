import os
from setuptools import find_packages, setup

import sys
import subprocess
from setuptools import Command

# Parameters -------------------------------------

IS_RELEASE = False
VERSION = '0.1.0'

parameters = {
    "NAME": "anomalous",
    "DESCRIPTION": "Time series outliers detection",
    "VERSION": VERSION,
    "REQUIREMENTS_FILES": ['requirements.txt'],
    "KEYWORDS": ('time series', 'outliers'),
    "IS_RELEASE": IS_RELEASE,
    "PACKAGES": find_packages(),
    "PACKAGE_DIRECTORY": os.path.dirname(os.path.abspath(__file__)),
    "LICENSE": 'GPL',
    "AUTHOR": "Jose Lopez",
}

# Code -------------------------------------


class BaseLintCommand(Command):
    """A setup.py lint subcommand developers can run locally."""
    description = "run code linter(s)"
    user_options = []
    initialize_options = finalize_options = lambda self: None
    package_name = ''
    working_directory = "."

    def run(self):
        """Lint current branch compared to a reasonable master branch"""
        sys.exit(subprocess.call(r'''
        set -eu
        upstream="$(git remote -v |
                    awk '/[@\/]github\.com[:\/lopezco\/%s[\. ]/{ print $1; exit }')"
        git fetch -q $upstream v3
        best_ancestor=$(git merge-base HEAD refs/remotes/$upstream/master)
        .travis/check_pylint_diff $best_ancestor
        ''' % self.package_name, shell=True, cwd=self.working_directory))


class BaseCoverageCommand(Command):
    """A setup.py coverage subcommand developers can run locally."""
    description = "run code coverage"
    user_options = []
    initialize_options = finalize_options = lambda self: None
    package_name = ''
    working_directory = "."

    def run(self):
        """Check coverage on current workdir"""
        sys.exit(subprocess.call(r'''
        coverage run --source=%s -m unittest -v {name}.tests
        echo; echo
        coverage report
        coverage html &&
            { echo; echo "See also: file://$(pwd)/htmlcov/index.html"; echo; }
        ''' % self.package_name, shell=True, cwd=self.working_directory))


def get_requirements_from_files(file_list, package_directory):
    return sorted(set(
        line.partition('#')[0].strip()
        for file in (os.path.join(package_directory, file)
                     for file in file_list)
        for line in open(file)
    ) - {''})


def git_version():
    """Return the git revision as a string.

    Copied from numpy setup.py
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


def write_version_py(version, package, is_release=False):
    # Copied from numpy setup.py
    filename = '{name}/version.py'.format(name=package)

    cnt = """
# THIS FILE IS GENERATED FROM {name} SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(is_release)s

if not release:
    version = full_version
    short_version += ".dev"
""".format(name=package.capitalize())

    FULLVERSION = version
    if os.path.exists('.git') or os.path.exists('../.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('{name}/version.py'.format(name=package)):
        # must be a source distribution, use existing version file
        import importlib
        _version = importlib.import_module("{name}.version".format(name=package), "{name}/version.py".format(name=package))

        GIT_REVISION = _version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not is_release:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': package,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'is_release': str(is_release)})
    finally:
        a.close()

    return FULLVERSION


_DEFAULT_CONFIG = {}


def setup_package(**kwargs):

    parameters = {}
    for key in _DEFAULT_CONFIG.keys():
        parameters[key] =  kwargs.get(key, _DEFAULT_CONFIG.get(key))

    readme_file = os.path.join(parameters["PACKAGE_DIRECTORY"], 'README.md')

    if os.path.exists(readme_file):
        long_description = open(readme_file).read()
    else:
        long_description = ""

    fullversion = write_version_py(version=parameters['VERSION'], package=parameters['NAME'],
                                                 is_release=parameters['IS_RELEASE'])

    class LintCommand(BaseLintCommand):
        package_name = parameters['NAME']
        working_directory = parameters['PACKAGE_DIRECTORY']

    class CoverageCommand(BaseCoverageCommand):
        package_name = parameters['NAME']
        working_directory = parameters['PACKAGE_DIRECTORY']

    cmdclass = {'lint': LintCommand, 'coverage': CoverageCommand}
    install_requires = get_requirements_from_files(parameters['REQUIREMENTS_FILES'], parameters['PACKAGE_DIRECTORY'])
    test_suite = '{name}.tests.suite'.format(name=parameters['NAME'])

    setup(
        name=parameters['NAME'],
        version=fullversion,
        description=parameters['DESCRIPTION'],
        long_description=long_description,
        author=parameters['AUTHOR'],
        author_email=parameters['AUTHOR_EMAIL'],
        url=parameters['URL'],
        license=parameters['LICENSE'],
        packages=parameters['PACKAGES'],
        package_data=parameters['PACKAGE_DATA'],
        install_requires=install_requires,
        zip_safe=False,
        test_suite=test_suite,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    setup_package(**parameters)
