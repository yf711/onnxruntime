// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class GridSampleOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status GridSampleOpBuilder::AddToModelBuilderImpl([[maybe_unused]] ModelBuilder& model_builder,
                                                  [[maybe_unused]] const Node& node,
                                                  [[maybe_unused]] const logging::Logger& logger) const {
#if defined(COREML_ENABLE_MLPROGRAM)
  using namespace CoreML::Specification::MILSpec;  // NOLINT
  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.resample

  auto op = model_builder.CreateOperation(node, "resample");

  NodeAttrHelper helper(node);
  const auto& mode = helper.Get("mode", "linear");
  const bool align_corners = helper.Get("align_corners", 0);
  const std::string coordinates_mode = "normalized_minus_one_to_one";
  std::string padding_mode = helper.Get("padding_mode", "zeros");
  if (padding_mode == "zeros") {
    padding_mode = "constant";
  }

  const auto input_defs = node.InputDefs();
  const auto output_defs = node.OutputDefs();
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationInput(*op, "coordinates", input_defs[1]->Name());
  AddOperationInput(*op, "sampling_mode", model_builder.AddScalarConstant(op->type(), "sampling_mode", mode));
  AddOperationInput(*op, "padding_mode", model_builder.AddScalarConstant(op->type(), "padding_mode", padding_mode));
  AddOperationInput(*op, "pading_value", model_builder.AddScalarConstant(op->type(), "padding_value", 0.0f));
  AddOperationInput(*op, "coordinates_mode",
                    model_builder.AddScalarConstant(op->type(), "coordinates_mode", coordinates_mode));
  AddOperationInput(*op, "align_corners", model_builder.AddScalarConstant(op->type(), "align_corners", align_corners));

  AddOperationOutput(*op, *output_defs[0]);
#endif
  return Status::OK();
}

bool GridSampleOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "GridSample is not supported.";
    return false;
  }

  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  const auto input_rank = input_shape.size();
  if (input_rank < 4) {
    LOGS(logger, VERBOSE) << "GridSample only supports 4D input not " << input_rank << "D";
  }

  NodeAttrHelper helper(node);

  const auto& mode = helper.Get("mode", "linear");
  if (mode == "cubic") {
    LOGS(logger, VERBOSE) << "GridSample does not support cubic interpolation";
    return false;
  }

  return true;
}

void CreateGridSampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GridSampleOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
