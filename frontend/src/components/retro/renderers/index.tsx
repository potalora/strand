"use client";

import React from "react";
import { getObservationCategory } from "./shared";
import { ConditionRenderer } from "./ConditionRenderer";
import { ObservationLabRenderer } from "./ObservationLabRenderer";
import { ObservationVitalRenderer } from "./ObservationVitalRenderer";
import { ObservationSocialRenderer } from "./ObservationSocialRenderer";
import { MedicationRenderer } from "./MedicationRenderer";
import { EncounterRenderer } from "./EncounterRenderer";
import { ImmunizationRenderer } from "./ImmunizationRenderer";
import { AllergyRenderer } from "./AllergyRenderer";
import { ProcedureRenderer } from "./ProcedureRenderer";
import { ServiceRequestRenderer } from "./ServiceRequestRenderer";
import { DocumentRenderer } from "./DocumentRenderer";
import { DiagnosticReportRenderer } from "./DiagnosticReportRenderer";
import { ImagingRenderer } from "./ImagingRenderer";
import { CarePlanRenderer } from "./CarePlanRenderer";
import { CommunicationRenderer } from "./CommunicationRenderer";
import { AppointmentRenderer } from "./AppointmentRenderer";
import { CareTeamRenderer } from "./CareTeamRenderer";
import { QuestionnaireResponseRenderer } from "./QuestionnaireResponseRenderer";
import { ImmunizationRecommendationRenderer } from "./ImmunizationRecommendationRenderer";
import { GenericRenderer } from "./GenericRenderer";

export interface FhirResourceRendererProps {
  recordType: string;
  fhirResource: Record<string, unknown>;
}

export function FhirResourceRenderer({
  recordType,
  fhirResource,
}: FhirResourceRendererProps) {
  const r = fhirResource;
  const type = recordType.toLowerCase();

  switch (type) {
    case "medication":
      return <MedicationRenderer r={r} />;

    case "condition":
      return <ConditionRenderer r={r} />;

    case "encounter":
      return <EncounterRenderer r={r} />;

    case "observation": {
      const category = getObservationCategory(r);
      if (category === "vital-signs") return <ObservationVitalRenderer r={r} />;
      if (category === "social-history") return <ObservationSocialRenderer r={r} />;
      return <ObservationLabRenderer r={r} />;
    }

    case "document":
      return <DocumentRenderer r={r} />;

    case "immunization":
      return <ImmunizationRenderer r={r} />;

    case "allergy":
      return <AllergyRenderer r={r} />;

    case "procedure":
      return <ProcedureRenderer r={r} />;

    case "service_request":
      return <ServiceRequestRenderer r={r} />;

    case "diagnostic_report":
      return <DiagnosticReportRenderer r={r} />;

    case "imaging":
      return <ImagingRenderer r={r} />;

    case "care_plan":
      return <CarePlanRenderer r={r} />;

    case "communication":
      return <CommunicationRenderer r={r} />;

    case "appointment":
      return <AppointmentRenderer r={r} />;

    case "care_team":
      return <CareTeamRenderer r={r} />;

    case "questionnaire_response":
      return <QuestionnaireResponseRenderer r={r} />;

    case "immunization_recommendation":
      return <ImmunizationRecommendationRenderer r={r} />;

    default:
      return <GenericRenderer r={r} />;
  }
}
