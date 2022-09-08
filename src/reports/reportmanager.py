import os
from abc import ABC, abstractstaticmethod
from datetime import datetime

from .reportsections import *

SECTIONS = {
    "config": ReportConfig,
    "data": ReportData,
    "model": ReportModel,
    "training": ReportTraining,
    "results": ReportResults,
    "footer": ReportFooter,
}


class ReportManager:
    """
    Report object which can store, print and save the sections listed above.

    This object essentially acts as a blank page which holds all the sections
    we are interested in.

    Methods:
    - add_section(): Adds in a specific section to the report which is then
                     held inside the ReportManager object.

    - save(): This saves sections we have added to a report and outputs these
              to the terminal.
    """

    def __init__(self, path_prefix):
        self._path_prefix = path_prefix
        self._sections = []

        if not self._path_prefix or not isinstance(self._path_prefix, str):
            raise ValueError(
                "Please input a valid path. Path must be a valid string.\n"
                + f"Current path: {self._path_prefix}"
            )

        dt = datetime.now().strftime("%m%d%Y_%H%M%S")
        if not os.path.exists(self._path_prefix):
            os.makedirs(self._path_prefix, mode=0o777)

        self._path = os.path.join(self._path_prefix, dt)

        os.mkdir(self._path, mode=0o777)

    @property
    def path(self):
        return self._path

    def add_section(self, section, **kwargs):
        new_section = SECTIONS[section.lower()](path=self._path, **kwargs)
        self._sections.append(new_section)

    def save(self):
        with open(os.path.join(self._path, "report.md"), "a") as report_file:
            for section in self._sections:
                print(str(section))
                report_file.write(str(section))
        self._sections = []
