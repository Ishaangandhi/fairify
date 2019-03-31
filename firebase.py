import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from fair_utils import *

class FirebaseModel:
    def __init__(self, dm):
        self.dm = dm
        # Fetch the service account key JSON file contents
        cred = credentials.Certificate(
            'fb-creds.json')

        # Initialize the app with a service account, granting admin privileges
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://lahacks2019-2ba9b.firebaseio.com/'
        })

        # As an admin, the app has access to read and write all data,
        # regradless of Security Rules
        self.interactive = db.reference('/').child("interactive")
        self.interactive.set({
            "a1": "",
            "a2": "",
            "b1": "",
            "b2": "",
            "biased": False
        })
        self.ref = db.reference('/').child("models")
        self.model_num = 0
        self.fb_model = self.ref.child(str(self.model_num))
        self.analogies = []
        self.analogies_ref = self.fb_model.child("analogies")
        self.analogies_ref.set({})

    # On save, get the code block and options, then push them to firebase
    def add_analogy(self, analogy, override=False):
        self.analogies.append(analogy)
        if len(self.analogies)<160 or override:
            self.analogies_ref.update({str(len(self.analogies)-1):self.analogies[-1]})# dict(enumerate(self.analogies)))


    def update_name(self, name):
        self.fb_model.update({"name": name})

    def update_percent(self, percent):
        self.fb_model.update({"percent": percent})

    def listen(self):
        """
            Watch for changes to the interactive-mode values (a1, )
        """
        self.interactive.child("a1").listen(self.new_analogy)
        self.interactive.child("a2").listen(self.new_analogy)
        self.interactive.child("b1").listen(self.new_analogy)
        self.interactive.child("biased").listen(self.new_biased_analogy)
        self.analogies_ref.listen(self.analogy_listener)

    def analogy_listener(self, _):
        """
            Analogy was marked as needing to be fixed. Neutralize it.
        """
        analogies = self.analogies_ref.get()
        if not analogies:
            return
        for i, analogy in enumerate(analogies):
            if analogy["should_fix"]:
                # print("fixing analogy!")
                triad = (analogy["a1"].lower(), analogy["a2"].lower(), analogy["b1"].lower())
                g = self.dm.word_to_vec_map[analogy["a1"].lower()] - self.dm.word_to_vec_map[analogy["b1"].lower()]
                self.dm.word_to_vec_map = correct_bias(triad, g, self.dm.word_to_vec_map)
                self.analogies[i]["is_fixed"] = True
                self.analogies_ref.update({str(i) : self.analogies[i]})
                break

    def new_biased_analogy(self, _):
        """
            Analogy was marked as biased. Add it to Firebase.
        """
        interactive = self.interactive.get()
        a1 = interactive["a1"].lower()
        a2 = interactive["a2"].lower()
        b1 = interactive["b1"].lower()
        completion = interactive["b2"].lower()
        biased = interactive["biased"]
        if not biased:
            return

        g = self.dm.word_to_vec_map[a1] - self.dm.word_to_vec_map[b1]
        e1 = neutralize(a2, g, self.dm.word_to_vec_map)
        e0, e2 = equalize((a1, b1), g, self.dm.word_to_vec_map)
        self.add_analogy(create_analogy((a1, a2, b1), completion, e0, e1, e2,
                    g, self.dm.word_to_vec_map, True), override=True)


    def new_analogy(self, _):
        """
            Show changes to the interactive-mode values
        """
        interactive = self.interactive.get()
        a1 = interactive["a1"]
        a2 = interactive["a2"]
        b1 = interactive["b1"]

        if not a1 or not a2 or not b1:
            # cleared entry, nothing to do
            return

        completion = complete_analogy(a1, a2, b1, self.dm.word_to_vec_map)
        interactive = self.interactive.update({"b2": completion})
