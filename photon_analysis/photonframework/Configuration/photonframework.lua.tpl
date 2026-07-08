conflict("photonframework")
help([[
Framework for AMS-02 gamma ray analysis
]])

prereq("foss/2024a")

if not ( isloaded("Python") ) then
    load("Python")
end
if not ( isloaded("SciPy-bundle") ) then
    load("SciPy-bundle")
end
if not ( isloaded("Python-bundle-PyPI") ) then
    load("PyPI-bundle")
end
if not ( isloaded("uproot") ) then
    load("uproot")
end
if not ( isloaded("matplotlib") ) then
    load("matplotlib")
end
if not ( isloaded("ACsoft") ) then
    try_load("ACsoft")
end
setenv("PHOTONFRAMEWORK", "%REPOSITORY_DIRECTORY%")
append_path("PATH", os.getenv("PHOTONFRAMEWORK") .. "/bin")
append_path("PATH", os.getenv("PHOTONFRAMEWORK") .. "/Scripts")
append_path("LD_LIBRARY_PATH", os.getenv("PHOTONFRAMEWORK") .. "/lib")
append_path("LIBRARY_PATH", os.getenv("PHOTONFRAMEWORK") .. "/lib")
append_path("PYTHONPATH", os.getenv("PHOTONFRAMEWORK") .. "/Scripts")
append_path("COOKBOOK_DNA", os.getenv("PHOTONFRAMEWORK") .. "/Configuration/cookbook.conf")
if ( isloaded("ACsoft") ) then
    append_path("CLUSTERS_JOB_SETTINGS", os.getenv("PHOTONFRAMEWORK") .. "/Configuration/analysis-job-settings.cfg")
else
    append_path("CLUSTERS_JOB_SETTINGS", os.getenv("PHOTONFRAMEWORK") .. "/Configuration/analysis-job-settings-no-acsoft.cfg")
end
setenv("VIRTUAL_ENV", "%VIRTUALENV_DIRECTORY%")
prepend_path("PATH", os.getenv("VIRTUAL_ENV") .. "/bin")
