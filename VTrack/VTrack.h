#pragma once

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the VTRACK_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// VTRACK_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef VTRACK_EXPORTS
#define VTRACK_API __declspec(dllexport)
#else
#define VTRACK_API __declspec(dllimport)
#endif

#include "pluginterfaces/base/fplatform.h"
#include "pluginterfaces/base/funknown.h"

static const Steinberg::FUID VTrackProcessorUID(0x7B586823, 0x8E484472, 0xB8D8AAA2, 0xC5941C12);
static const Steinberg::FUID VTrackControllerUID(0xA3906BF7, 0x497546EF, 0xAFB447B4, 0x473B0263);
