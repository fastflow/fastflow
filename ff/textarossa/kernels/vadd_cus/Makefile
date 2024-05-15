# Options for TARGET: sw_emu, hw_emu and hw
TARGET ?= sw_emu
DEVICE ?= u50
PLATFORM ?= xilinx_$(DEVICE)_gen3x16_xdma_5_202210_1

PLATFORM_REPO_PATHS ?= /opt/xilinx/platforms
PFM := $(PLATFORM_REPO_PATHS)/$(PLATFORM)/$(PLATFORM).xpfm

VPP = v++
VPPINCLUDES := -I.
VPPFLAGS := --platform $(PFM) -t $(TARGET) -s -g
VPPCONNECT := --config connectivity.ini


KERNEL_NAME := vadd
KERNEL_FILE := $(KERNEL_NAME).cpp
KERNEL_XO := $(KERNEL_NAME).xo
KERNEL_XCLBIN := $(KERNEL_NAME).xclbin


$(KERNEL_XO): $(KERNEL_FILE)
	$(VPP) -k $(KERNEL_NAME) $(VPPFLAGS) $(VPPINCLUDES) $(VPPCONNECT) -c -o $@ $<

$(KERNEL_XCLBIN): $(KERNEL_XO)
	$(VPP) -l $(VPPFLAGS) $(VPPINCLUDES) $(VPPCONNECT) -o $@ $<

emconfig: emconfig.json
emconfig.json:
	emconfigutil --platform $(PLATFORM)


all: emconfig $(KERNEL_XCLBIN)

cleanall: clean
clean:
	$(RM) -rf *.xo _x .run .Xil *.xclbin *.ltx *.log *.info *compile_summary* vitis_analyzer* *link_summary* *.o main emconfig.json device_trace_0.csv *.xclbin.package_summary opencl_trace.csv xrt.run_summary summary.csv
