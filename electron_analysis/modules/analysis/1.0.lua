conflict("analysis")
help([[
Electron Analysis PhD project
]])

if not ( isloaded("Python") ) then
    load("Python")
end
if not ( isloaded("numpy") ) then
    load("numpy")
end
if not ( isloaded("scipy") ) then
    load("scipy")
end
if not ( isloaded("uproot") ) then
    load("uproot")
end
if not ( isloaded("matplotlib") ) then
    load("matplotlib")
end
if not ( isloaded("ACsoft") ) then
    load("ACsoft")
end
setenv("MY_ANALYSIS", "/home/op115134/Software/YasamanAnalysis")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/bin")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/template_fit")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/Cuts")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/flux")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/tools")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/Histograms/Estimators")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/Histograms/IdentificationCuts")
append_path("PATH", os.getenv("MY_ANALYSIS") .. "/Scripts/Histograms/SelectionCuts")
append_path("LD_LIBRARY_PATH", os.getenv("MY_ANALYSIS") .. "/lib")
append_path("LIBRARY_PATH", os.getenv("MY_ANALYSIS") .. "/lib")
append_path("PYTHONPATH", os.getenv("MY_ANALYSIS") .. "/Scripts")
append_path("CLUSTERS_JOB_SETTINGS", os.getenv("MY_ANALYSIS") .. "/Configuration/analysis-job-settings.cfg")
setenv("VIRTUAL_ENV", "/home/rs429310/Software/venv")
prepend_path("PATH", os.getenv("VIRTUAL_ENV") .. "/bin")
