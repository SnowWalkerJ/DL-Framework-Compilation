SOURCE_DIR=sources
GRAPH_SOURCES=$(wildcard $(SOURCE_DIR)/*.dot)
BUILD_DIR=build
GRAPH_BUILD_DIR=$(BUILD_DIR)/graph
GRAPH_BUILD_IMAGES=$(addprefix $(GRAPH_BUILD_DIR)/, $(GRAPH_SOURCES:sources/%.dot=%.png))

$(GRAPH_BUILD_DIR)/%.png: $(SOURCE_DIR)/%.dot
	mkdir -p $(GRAPH_BUILD_DIR)
	dot $< -Tpng -o $@

$(BUILD_DIR)/main.pdf: main.md $(GRAPH_BUILD_IMAGES)
	pandoc main.md -o $(BUILD_DIR)/main.pdf

$(BUILD_DIR)/pre.pdf: pre.md $(GRAPH_BUILD_IMAGES)
	pandoc pre.md -o $(BUILD_DIR)/pre.pdf

pdf: $(BUILD_DIR)/main.pdf $(BUILD_DIR)/pre.pdf

clean:
	rm -r build

.PHONY: pdf clean
