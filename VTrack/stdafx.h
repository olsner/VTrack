// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define NOMINMAX
// Windows Header Files:
#include <windows.h>

#include <assert.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
using std::min;
using std::max;
using std::make_shared;
using std::shared_ptr;

#include "pluginterfaces/base/fplatform.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "public.sdk/source/main/pluginfactoryvst3.h"
#include "public.sdk/source/vst/vstaudioeffect.h"
#include "public.sdk/source/vst/vsteditcontroller.h"
#include "public.sdk/source/vst/hosting/eventlist.h"

#include "vstgui4/vstgui/plugin-bindings/vst3editor.h"

using Steinberg::FUnknown;
using Steinberg::FUID;
using Steinberg::tresult;

#include "Vtrack.h"

static void Debug(const char *fmt, ...) {
	va_list ap, ap2;
	va_start(ap, &fmt);
	va_copy(ap2, ap);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	char buf[1024];
	vsnprintf(buf, sizeof(buf), fmt, ap2);
	OutputDebugStringA(buf);
	va_end(ap2);
}

#define HERE Debug("HERE: %s\n", __FUNCTION__)