{
  cout << " ++++ ExampleAnalysis rootlogon ++++ " << endl;
  gInterpreter->AddIncludePath("$MY_ANALYSIS/Libraries/ExampleAnalysisTree");
  gSystem->Load("$MY_ANALYSIS/lib/libExampleAnalysisLibrary");
}
