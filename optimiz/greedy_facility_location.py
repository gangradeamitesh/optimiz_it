from .facility_location import FacilityLocation

class GreedyFacilityLocation:

    def __init__(self , dataset , max_subset_size) -> None:
        self.dataset = dataset
        self.max_subset_size = max_subset_size
        self.facility_locatiion_model = FacilityLocation(dataset)

    def select_subset(self):

        selected_subset_indices = []
        curr_value = 0

        for _ in range(self.max_subset_size):
            best_candidate = None
            best_candidate_value = -float('inf')

            for i in range(self.dataset.shape[0]):
                if i not in selected_subset_indices:
                    