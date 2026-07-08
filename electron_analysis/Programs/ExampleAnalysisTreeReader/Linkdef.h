#ifndef Linkdef_h
#define Linkdef_h

#ifdef __CINT__

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedefs;

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// clang-format off
#pragma link C++ class ExampleAnalysisTreeReader+;
// clang-format on

#endif

#endif
