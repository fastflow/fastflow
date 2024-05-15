# https://github.com/Xilinx/Vitis_Accel_Examples/blob/master/hello_world/makefile_us_alveo.mk
# Look at the above link to complete the Makefile properly

# Options for TARGET: sw_emu, hw_emu and hw
TARGET ?= hw
# Options for DEVICE: u200, u250. Default platform is XDMA, defined by PLATFORM
DEVICE ?= u50
# If other some specific platforms needs to be used, provide platform path directly
PLATFORM ?= xilinx_$(DEVICE)_gen3x16_xdma_5_202210_1
# If your platform is not in the standard install area edit this line
PLATFORM_REPO_PATHS ?= /opt/xilinx/platforms
PFM := $(PLATFORM_REPO_PATHS)/$(PLATFORM)/$(PLATFORM).xpfm

BOARD_CONFIG := connectivity.ini
VPPINCLUDES := -I.

VPPFLAGS := --platform $(PFM) -t $(TARGET) --save-temps
VPPLFLAGS := # --config $(BOARD_CONFIG)

KERNEL_NAME := DeflateKernel
KERNEL_FILE := $(KERNEL_NAME).cpp
KERNEL_XO := $(KERNEL_NAME).xo
KERNEL_XCLBIN := $(KERNEL_NAME).xclbin
VPP = v++

all: $(KERNEL_XO) $(KERNEL_XCLBIN) emconfig

$(KERNEL_XO): $(KERNEL_FILE)
	$(VPP) -c --kernel $(KERNEL_NAME) $(VPPFLAGS) $(VPPINCLUDES) -o $@ $<

$(KERNEL_XCLBIN): $(KERNEL_XO)
	$(VPP) -l $(VPPFLAGS) $(VPPLFLAGS) -o $@ $<

emconfig:emconfig.json
emconfig.json:
	emconfigutil --platform $(PLATFORM)

clean:
	$(RM) -rf *.xo _x .Xil *.xclbin *.ltx *.log *.info *compile_summary* vitis_analyzer* *link_summary* .run emconfig.json
