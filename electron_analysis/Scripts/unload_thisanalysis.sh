# Source this script to remove paths specific to this ExampleAnalysis from the environment.

# Put the name of the environment variable pointing to your analysis repository here, if you chose
# to use a different variable than "MY_ANALYSIS" in the thisanalysis.sh script:
if [ -n "${MY_ANALYSIS}" ] ; then
   old_analysis=${MY_ANALYSIS}
fi
unset MY_ANALYSIS


drop_from_path()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "drop_from_path: needs 2 arguments"
      return 1
   fi

   p=$1
   drop=$2

   newpath=`echo $p | sed -e "s;:${drop}:;:;g" \
                          -e "s;:${drop};;g"   \
                          -e "s;${drop}:;;g"   \
                          -e "s;${drop};;g"`
}


if [ -n "${old_analysis}" ] ; then
   if [ -n "${PATH}" ]; then
      drop_from_path ${PATH} ${old_analysis}/bin
      PATH=$newpath
      drop_from_path ${PATH} ${old_analysis}/Scripts
      PATH=$newpath
   fi
   if [ -n "${LD_LIBRARY_PATH}" ]; then
      drop_from_path ${LD_LIBRARY_PATH} ${old_analysis}/lib
      LD_LIBRARY_PATH=$newpath
   fi
   if [ -n "${DYLD_LIBRARY_PATH}" ]; then
      drop_from_path ${DYLD_LIBRARY_PATH} ${old_analysis}/lib
      DYLD_LIBRARY_PATH=$newpath
   fi
   if [ -n "${SHLIB_PATH}" ]; then
      drop_from_path ${SHLIB_PATH} ${old_analysis}/lib
      SHLIB_PATH=$newpath
   fi
   if [ -n "${LIBPATH}" ]; then
      drop_from_path ${LIBPATH} ${old_analysis}/lib
      LIBPATH=$newpath
   fi
   if [ -n "${AC_COOKBOOK_MACRO_PATH}" ]; then
      drop_from_path ${AC_COOKBOOK_MACRO_PATH} ${old_analysis}
      AC_COOKBOOK_MACRO_PATH=$newpath
   fi
   if [ -n "${CLUSTERS_JOB_SETTINGS}" ]; then
      drop_from_path ${CLUSTERS_JOB_SETTINGS} ${old_analysis}/Configuration/analysis-job-settings.cfg
      CLUSTERS_JOB_SETTINGS=$newpath
   fi
   if [ -n "${ACSOFT_ADDITIONAL_LOGONS}" ]; then
      drop_from_path ${ACSOFT_ADDITIONAL_LOGONS} ${old_analysis}/rootlogon.C
      ACSOFT_ADDITIONAL_LOGONS=$newpath
   fi
   if [ -n "${AC_CHOOSE_UNLOAD_SCRIPTS}" ]; then
       drop_from_path ${AC_CHOOSE_UNLOAD_SCRIPTS} ${old_analysis}/Scripts/unload_thisanalysis.sh
       AC_CHOOSE_UNLOAD_SCRIPTS=$newpath
   fi
fi

unset old_analysis
