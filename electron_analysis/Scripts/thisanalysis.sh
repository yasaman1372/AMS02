# Source this script to set up the ExampleAnalysis build that this script is part of.

if [ -f "${MY_ANALYSIS}/Scripts/unload_thisanalysis.sh" ]; then
  source ${MY_ANALYSIS}/Scripts/unload_thisanalysis.sh
fi

if [ "x${BASH_ARGV[0]}" = "x" ]; then
    if [ ! -f Scripts/thisanalysis.sh ]; then
        echo ERROR: must "'cd where/analysis/is'" before calling "'source Scripts/thisanalysis.sh'" for this version of bash!
        MY_ANALYSIS=; export MY_ANALYSIS
        return 1
    fi
    MY_ANALYSIS="$PWD"; export MY_ANALYSIS
else
    # get param to "."
    thisanalysis=$(dirname ${BASH_ARGV[0]})
    MY_ANALYSIS=$(cd ${thisanalysis}/..;pwd); export MY_ANALYSIS
fi


if [ -z "${PATH}" ]; then
   PATH=${MY_ANALYSIS}/bin:${MY_ANALYSIS}/Scripts:${MY_ANALYSIS}/Scripts/Cuts:${MY_ANALYSIS}/Scripts/flux:${MY_ANALYSIS}/Scripts/template_fit:${MY_ANALYSIS}/Scripts/Histograms/Estimators:${MY_ANALYSIS}/Scripts/Histograms/IdentificationCuts/electron:${MY_ANALYSIS}/Scripts/Histograms/SelectionCuts/electron:${MY_ANALYSIS}/Scripts/example_scripts; export PATH
else
   PATH=${MY_ANALYSIS}/bin:${MY_ANALYSIS}/Scripts:${MY_ANALYSIS}/Scripts/Cuts:${MY_ANALYSIS}/Scripts/flux:${MY_ANALYSIS}/Scripts/template_fit:${MY_ANALYSIS}/Scripts/Histograms/Estimators:${MY_ANALYSIS}/Scripts/Histograms/IdentificationCuts/electron:${MY_ANALYSIS}/Scripts/Histograms/SelectionCuts/electron:${MY_ANALYSIS}/Scripts/example_scripts:${PATH}; export PATH 

fi

if [ -z "${LD_LIBRARY_PATH}" ]; then
   LD_LIBRARY_PATH=${MY_ANALYSIS}/lib; export LD_LIBRARY_PATH       # Linux, ELF HP-UX
else
   LD_LIBRARY_PATH=${MY_ANALYSIS}/lib:${LD_LIBRARY_PATH}; export LD_LIBRARY_PATH
fi

if [ -z "${DYLD_LIBRARY_PATH}" ]; then
   DYLD_LIBRARY_PATH=${MY_ANALYSIS}/lib; export DYLD_LIBRARY_PATH   # Mac OS X
else
   DYLD_LIBRARY_PATH=${MY_ANALYSIS}/lib:${DYLD_LIBRARY_PATH}; export DYLD_LIBRARY_PATH
fi

if [ -z "${SHLIB_PATH}" ]; then
   SHLIB_PATH=${MY_ANALYSIS}/lib; export SHLIB_PATH                 # legacy HP-UX
else
   SHLIB_PATH=${MY_ANALYSIS}/lib:${SHLIB_PATH}; export SHLIB_PATH
fi

if [ -z "${LIBPATH}" ]; then
   LIBPATH=${MY_ANALYSIS}/lib; export LIBPATH                       # AIX
else
   LIBPATH=${MY_ANALYSIS}/lib:${LIBPATH}; export LIBPATH
fi

if [ -z "${AC_COOKBOOK_MACRO_PATH}" ]; then
   AC_COOKBOOK_MACRO_PATH=${MY_ANALYSIS}; export AC_COOKBOOK_MACRO_PATH
else
   AC_COOKBOOK_MACRO_PATH=${MY_ANALYSIS}:${AC_COOKBOOK_MACRO_PATH}; export AC_COOKBOOK_MACRO_PATH
fi

if [ -z "${CLUSTERS_JOB_SETTINGS}" ]; then
   CLUSTERS_JOB_SETTINGS=${MY_ANALYSIS}/Configuration/analysis-job-settings.cfg; export CLUSTERS_JOB_SETTINGS
else
   CLUSTERS_JOB_SETTINGS=${CLUSTERS_JOB_SETTINGS}:${MY_ANALYSIS}/Configuration/analysis-job-settings.cfg; export CLUSTERS_JOB_SETTINGS
fi

if [ -z "${ACSOFT_ADDITIONAL_LOGONS}" ]; then
   ACSOFT_ADDITIONAL_LOGONS=${MY_ANALYSIS}/rootlogon.C; export ACSOFT_ADDITIONAL_LOGONS
else
   ACSOFT_ADDITIONAL_LOGONS=${ACSOFT_ADDITIONAL_LOGONS}:${MY_ANALYSIS}/rootlogon.C; export ACSOFT_ADDITIONAL_LOGONS
fi

if [ -z "${AC_CHOOSE_UNLOAD_SCRIPTS}" ]; then
    AC_CHOOSE_UNLOAD_SCRIPTS=${MY_ANALYSIS}/Scripts/unload_thisanalysis.sh; export AC_CHOOSE_UNLOAD_SCRIPTS
else
    AC_CHOOSE_UNLOAD_SCRIPTS=${AC_CHOOSE_UNLOAD_SCRIPTS}:${MY_ANALYSIS}/Scripts/unload_thisanalysis.sh; export AC_CHOOSE_UNLOAD_SCRIPTS
fi

export PYTHONPATH=$PYTHONPATH:$MY_ANALYSIS/Scripts

unset thisanalysis
