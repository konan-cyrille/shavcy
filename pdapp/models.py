from pdapp import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(50), nullable=False)
    pseudo = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return '<User {}>'.fromat(self.nom)
        #self.pseudo = pseudo


class Images(db.Model):
    img_id = db.Column(db.Integer, primary_key=True)
    url_img = db.Column(db.String(250), nullable=False)
    nom_plante = db.Column(db.String(50), nullable=False)
    annotation = db.relationship('Annotations', backref='images', lazy=True)

    def __repr__(self):
        return '<Images {}>'.fromat(self.url_img)


class Annotations(db.Model):
    annot_id = db.Column(db.Integer, primary_key=True)
    url_img_annoted = db.Column(db.String(250), nullable=False)
    img_id = db.Column(db.Integer, db.ForeignKey('images.img_id'), nullable=False)

    def __repr__(self):
        return '<Annotations {}>'.fromat(self.url_img_annoted)

