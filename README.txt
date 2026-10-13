SheafCode source-code archive, 13 October 2026

Source comments and Python docstrings have been removed from these copies.
Executable strings, including generated-document text, are retained.
Original working source files have not been changed.

Contents:
Research_Release: model code, acquisition, experiments, extensions and tests.
audit_tools: project verification and document-building scripts.
SUBMISSION_WORKSPACE: manuscript audits, figure and presentation scripts.
Final_Manuscript and manuscript: available figure/manuscript tools.
Root launchers: original CMD and PowerShell entry points.
Dependency files are retained in their original folders.

This is a comprehensive code archive. It also includes historical and
exploratory scripts; not every script contributes to the final manuscript.
Folder paths are preserved for local imports and path-relative lookups.
Monitoring data, saved model/results files, installed dependencies, Python
executables, TeX executables, caches and credentials/configuration files
are not included. Scripts requiring those inputs must be provided with them.
Windows launchers expect the original audit_tools/python installation.

MANIFEST.json records original and packaged hashes and validation results.
Removing comments changes source hashes and removes Python __doc__ values.
Frozen research manifests still refer to the original-source hashes.
Python code was compiled and compared with its original abstract syntax
structure after removing docstrings. PowerShell was parsed and executable
tokens compared. R source was checked lexically; an R runtime was unavailable.
The complete research pipeline was not rerun for this packaging task.

Third-party attribution:
GRIN is by Andrea Cini, Ivan Marisca and Cesare Alippi.
The vendored original README and SOURCE.json identify its repository and
pinned commit. Their original hash inventory describes the unstripped code.
The three imported R files are from the openair project, retained under
Research_Release/uk_validation_results/raw.
LEGAL_NOTICES.json preserves copyright/SPDX text found in the source.
